#!/bin/bash
#
# ACE-Step Music Generation CLI (Bash + Curl)
#
# Requirements: curl (no jq needed)
#
# Usage:
#   ./acemusic.sh generate "Music description" [options]
#   ./acemusic.sh random [--no-thinking]
#   ./acemusic.sh status <job_id>
#   ./acemusic.sh models
#   ./acemusic.sh health
#   ./acemusic.sh config [--get|--set|--reset]
#
# Output:
#   - Results saved to output/<job_id>.json
#   - Audio files downloaded to output/<job_id>_1.mp3, output/<job_id>_2.mp3, ...

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/config.json"
# Output dir at same level as .claude (go up 4 levels from scripts/)
OUTPUT_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd)/acemusic_output"
DEFAULT_API_URL="http://127.0.0.1:8001"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Check dependencies
check_deps() {
    if ! command -v curl &> /dev/null; then
        echo -e "${RED}Error: curl is required but not installed.${NC}"
        exit 1
    fi
}

# Simple JSON value extractor (no jq needed)
# Usage: json_get "$json" "key"
json_get() {
    local json="$1"
    local key="$2"
    # Handle nested keys like "generation.thinking"
    if [[ "$key" == *.* ]]; then
        local first="${key%%.*}"
        local rest="${key#*.}"
        # Extract the nested object
        local nested=$(echo "$json" | sed 's/.*"'"$first"'"[[:space:]]*:[[:space:]]*{/{/' | sed 's/}.*/}/' | head -1)
        json_get "$nested" "$rest"
    else
        # Extract string value
        local result=$(echo "$json" | grep -o '"'"$key"'"[[:space:]]*:[[:space:]]*"[^"]*"' | sed 's/.*:[[:space:]]*"\([^"]*\)".*/\1/' | head -1)
        if [ -z "$result" ]; then
            # Try non-string value (number, boolean, null)
            result=$(echo "$json" | grep -o '"'"$key"'"[[:space:]]*:[[:space:]]*[^,}]*' | sed 's/.*:[[:space:]]*//' | tr -d ' ' | head -1)
        fi
        echo "$result"
    fi
}

# Extract array values (for audio_paths)
json_get_array() {
    local json="$1"
    local key="$2"
    echo "$json" | grep -o '"'"$key"'"[[:space:]]*:[[:space:]]*\[[^]]*\]' | sed 's/.*\[//;s/\].*//' | tr ',' '\n' | sed 's/^[[:space:]]*"//;s/"[[:space:]]*$//'
}

# Escape string for JSON
json_escape() {
    local str="$1"
    # Escape backslashes, double quotes, and control characters
    str="${str//\\/\\\\}"
    str="${str//\"/\\\"}"
    str="${str//$'\n'/\\n}"
    str="${str//$'\r'/\\r}"
    str="${str//$'\t'/\\t}"
    echo "$str"
}

# Ensure output directory exists
ensure_output_dir() {
    mkdir -p "$OUTPUT_DIR"
}

# Default config
DEFAULT_CONFIG='{
  "api_url": "http://127.0.0.1:8001",
  "generation": {
    "thinking": true,
    "use_format": true,
    "use_cot_caption": true,
    "use_cot_language": true,
    "audio_format": "mp3",
    "vocal_language": "en"
  }
}'

# Ensure config file exists
ensure_config() {
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "$DEFAULT_CONFIG" > "$CONFIG_FILE"
    fi
}

# Get config value
get_config() {
    local key="$1"
    ensure_config
    local json=$(cat "$CONFIG_FILE" | tr -d '\n')
    json_get "$json" "$key"
}

# Set config value (simple implementation)
set_config() {
    local key="$1"
    local value="$2"
    ensure_config

    local tmp_file="${CONFIG_FILE}.tmp"

    if [[ "$key" == *.* ]]; then
        # Nested key like "generation.thinking"
        local subkey="${key#*.}"

        if [ "$value" = "true" ] || [ "$value" = "false" ]; then
            sed "s/\"$subkey\"[[:space:]]*:[[:space:]]*[^,}]*/\"$subkey\": $value/" "$CONFIG_FILE" > "$tmp_file"
        elif [[ "$value" =~ ^[0-9]+$ ]] || [[ "$value" =~ ^[0-9]+\.[0-9]+$ ]]; then
            sed "s/\"$subkey\"[[:space:]]*:[[:space:]]*[^,}]*/\"$subkey\": $value/" "$CONFIG_FILE" > "$tmp_file"
        else
            sed "s/\"$subkey\"[[:space:]]*:[[:space:]]*\"[^\"]*\"/\"$subkey\": \"$value\"/" "$CONFIG_FILE" > "$tmp_file"
        fi
    else
        # Top-level key
        if [ "$value" = "true" ] || [ "$value" = "false" ]; then
            sed "s/\"$key\"[[:space:]]*:[[:space:]]*[^,}]*/\"$key\": $value/" "$CONFIG_FILE" > "$tmp_file"
        elif [[ "$value" =~ ^[0-9]+$ ]] || [[ "$value" =~ ^[0-9]+\.[0-9]+$ ]]; then
            sed "s/\"$key\"[[:space:]]*:[[:space:]]*[^,}]*/\"$key\": $value/" "$CONFIG_FILE" > "$tmp_file"
        else
            sed "s/\"$key\"[[:space:]]*:[[:space:]]*\"[^\"]*\"/\"$key\": \"$value\"/" "$CONFIG_FILE" > "$tmp_file"
        fi
    fi

    mv "$tmp_file" "$CONFIG_FILE"
    echo "Set $key = $value"
}

# Load API URL
load_api_url() {
    local url=$(get_config "api_url")
    echo "${url:-$DEFAULT_API_URL}"
}

# Check API health
check_health() {
    local url="$1"
    local status
    status=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 5 "${url}/health" 2>/dev/null) || true
    [ "$status" = "200" ]
}

# Prompt for URL
prompt_for_url() {
    echo ""
    echo -e "${YELLOW}API server is not responding.${NC}"
    echo "Please enter the API URL (or press Enter for default):"
    read -p "API URL [$DEFAULT_API_URL]: " user_input
    echo "${user_input:-$DEFAULT_API_URL}"
}

# Ensure API connection
ensure_connection() {
    ensure_config
    local api_url=$(load_api_url)

    if check_health "$api_url"; then
        echo "$api_url"
        return 0
    fi

    echo -e "${YELLOW}Cannot connect to: $api_url${NC}" >&2
    local new_url=$(prompt_for_url)

    if check_health "$new_url"; then
        set_config "api_url" "$new_url" > /dev/null
        echo -e "${GREEN}Saved API URL: $new_url${NC}" >&2
        echo "$new_url"
        return 0
    fi

    echo -e "${RED}Error: Cannot connect to $new_url${NC}" >&2
    exit 1
}

# Download audio files from result
download_audios() {
    local api_url="$1"
    local job_id="$2"
    local result_json="$3"

    ensure_output_dir

    local audio_format=$(get_config "generation.audio_format")
    [ -z "$audio_format" ] && audio_format="mp3"

    local count=1
    for audio_path in $(json_get_array "$result_json" "audio_paths"); do
        if [ -n "$audio_path" ]; then
            local output_file="${OUTPUT_DIR}/${job_id}_${count}.${audio_format}"
            local download_url="${api_url}${audio_path}"

            echo -e "  ${CYAN}Downloading audio $count...${NC}"
            if curl -s -o "$output_file" "$download_url"; then
                echo -e "  ${GREEN}Saved: $output_file${NC}"
            else
                echo -e "  ${RED}Failed to download: $download_url${NC}"
            fi
            count=$((count + 1))
        fi
    done
}

# Save result to JSON file
save_result() {
    local job_id="$1"
    local result_json="$2"

    ensure_output_dir
    local output_file="${OUTPUT_DIR}/${job_id}.json"
    echo "$result_json" > "$output_file"
    echo -e "${GREEN}Result saved: $output_file${NC}"
}

# Health command
cmd_health() {
    check_deps
    ensure_config
    local api_url=$(load_api_url)

    echo "Checking API at: $api_url"
    if check_health "$api_url"; then
        echo -e "${GREEN}Status: OK${NC}"
        curl -s "${api_url}/health"
        echo ""
    else
        echo -e "${RED}Status: FAILED${NC}"
        exit 1
    fi
}

# Config command
cmd_config() {
    check_deps
    ensure_config

    local action=""
    local key=""
    local value=""

    while [[ $# -gt 0 ]]; do
        case $1 in
            --get) action="get"; key="$2"; shift 2 ;;
            --set) action="set"; key="$2"; value="$3"; shift 3 ;;
            --reset) action="reset"; shift ;;
            --list) action="list"; shift ;;
            *) shift ;;
        esac
    done

    case "$action" in
        "get")
            [ -z "$key" ] && { echo -e "${RED}Error: --get requires KEY${NC}"; exit 1; }
            local result=$(get_config "$key")
            [ -n "$result" ] && echo "$key = $result" || echo "Key not found: $key"
            ;;
        "set")
            [ -z "$key" ] || [ -z "$value" ] && { echo -e "${RED}Error: --set requires KEY VALUE${NC}"; exit 1; }
            set_config "$key" "$value"
            ;;
        "reset")
            echo "$DEFAULT_CONFIG" > "$CONFIG_FILE"
            echo -e "${GREEN}Configuration reset to defaults.${NC}"
            cat "$CONFIG_FILE"
            ;;
        "list")
            echo "Current configuration:"
            cat "$CONFIG_FILE"
            ;;
        *)
            echo "Config file: $CONFIG_FILE"
            echo "Output dir: $OUTPUT_DIR"
            echo "----------------------------------------"
            cat "$CONFIG_FILE"
            echo "----------------------------------------"
            echo ""
            echo "Usage:"
            echo "  config --list              Show config"
            echo "  config --get <key>         Get value"
            echo "  config --set <key> <val>   Set value"
            echo "  config --reset             Reset to defaults"
            ;;
    esac
}

# Models command
cmd_models() {
    check_deps
    local api_url=$(ensure_connection)

    echo "Available Models:"
    echo "----------------------------------------"
    curl -s "${api_url}/v1/models"
    echo ""
}

# Status command
cmd_status() {
    check_deps
    local job_id="$1"

    [ -z "$job_id" ] && { echo -e "${RED}Error: job_id required${NC}"; echo "Usage: $0 status <job_id>"; exit 1; }

    local api_url=$(ensure_connection)
    local response=$(curl -s "${api_url}/v1/jobs/${job_id}")

    local status=$(json_get "$response" "status")
    echo "Job ID: $(json_get "$response" "job_id")"
    echo "Status: $status"

    if [ "$status" = "queued" ]; then
        echo "Queue Position: $(json_get "$response" "queue_position")"
    fi

    if [ "$status" = "succeeded" ]; then
        echo ""
        echo "Result:"
        echo "  BPM: $(json_get "$response" "bpm")"
        echo "  Key: $(json_get "$response" "keyscale")"
        echo "  Duration: $(json_get "$response" "duration")s"

        # Save and download
        save_result "$job_id" "$response"
        download_audios "$api_url" "$job_id" "$response"
    fi

    if [ "$status" = "failed" ]; then
        echo ""
        echo -e "${RED}Error: $(json_get "$response" "error")${NC}"
    fi
}

# Wait for job and download results
wait_for_job() {
    local api_url="$1"
    local job_id="$2"

    echo "Job created: $job_id"
    echo "Output: $OUTPUT_DIR"
    echo ""

    while true; do
        local response=$(curl -s "${api_url}/v1/jobs/${job_id}")
        local status=$(json_get "$response" "status")

        case "$status" in
            "succeeded")
                echo ""
                echo -e "${GREEN}Generation completed!${NC}"
                echo ""
                echo "Metadata:"
                echo "  BPM: $(json_get "$response" "bpm")"
                echo "  Key: $(json_get "$response" "keyscale")"
                echo "  Duration: $(json_get "$response" "duration")s"
                echo ""

                # Save result JSON
                save_result "$job_id" "$response"

                # Download audio files
                echo "Downloading audio files..."
                download_audios "$api_url" "$job_id" "$response"

                echo ""
                echo -e "${GREEN}Done! Files saved to: $OUTPUT_DIR${NC}"
                return 0
                ;;
            "failed")
                echo ""
                echo -e "${RED}Generation failed!${NC}"
                echo "Error: $(json_get "$response" "error")"

                # Save error result
                save_result "$job_id" "$response"
                return 1
                ;;
            "queued")
                local pos=$(json_get "$response" "queue_position")
                printf "\rQueued (position: %s)...    " "${pos:-?}"
                ;;
            *)
                printf "\rGenerating...              "
                ;;
        esac
        sleep 5
    done
}

# Generate command
cmd_generate() {
    check_deps
    ensure_config

    local caption="" lyrics="" description="" thinking="" use_format=""
    local no_thinking=false no_format=false no_wait=false
    local model="" language="" steps="" guidance="" seed="" duration="" bpm="" batch=""

    while [[ $# -gt 0 ]]; do
        case $1 in
            --caption|-c) caption="$2"; shift 2 ;;
            --lyrics|-l) lyrics="$2"; shift 2 ;;
            --description|-d) description="$2"; shift 2 ;;
            --thinking|-t) thinking="true"; shift ;;
            --no-thinking) no_thinking=true; shift ;;
            --use-format) use_format="true"; shift ;;
            --no-format) no_format=true; shift ;;
            --model|-m) model="$2"; shift 2 ;;
            --language) language="$2"; shift 2 ;;
            --steps) steps="$2"; shift 2 ;;
            --guidance) guidance="$2"; shift 2 ;;
            --seed) seed="$2"; shift 2 ;;
            --duration) duration="$2"; shift 2 ;;
            --bpm) bpm="$2"; shift 2 ;;
            --batch) batch="$2"; shift 2 ;;
            --no-wait) no_wait=true; shift ;;
            *) [ -z "$caption" ] && caption="$1"; shift ;;
        esac
    done

    # If no caption but has description, use simple mode
    if [ -z "$caption" ] && [ -z "$description" ]; then
        echo -e "${RED}Error: caption or description required${NC}"
        echo "Usage: $0 generate \"Music description\" [options]"
        echo "       $0 generate -d \"Simple description\" [options]"
        exit 1
    fi

    local api_url=$(ensure_connection)

    # Get defaults
    local def_thinking=$(get_config "generation.thinking")
    local def_format=$(get_config "generation.use_format")
    local def_cot_caption=$(get_config "generation.use_cot_caption")
    local def_cot_language=$(get_config "generation.use_cot_language")
    local def_language=$(get_config "generation.vocal_language")
    local def_audio_format=$(get_config "generation.audio_format")

    [ -z "$thinking" ] && thinking="${def_thinking:-true}"
    [ -z "$use_format" ] && use_format="${def_format:-true}"
    [ -z "$language" ] && language="${def_language:-en}"

    [ "$no_thinking" = true ] && thinking="false"
    [ "$no_format" = true ] && use_format="false"

    # Build payload manually (no jq) - escape strings for JSON
    local esc_caption=$(json_escape "$caption")
    local esc_lyrics=$(json_escape "${lyrics:-}")
    local esc_description=$(json_escape "${description:-}")

    local payload="{"
    payload+="\"caption\":\"${esc_caption}\","
    payload+="\"lyrics\":\"${esc_lyrics}\","
    payload+="\"sample_query\":\"${esc_description}\","
    payload+="\"thinking\":${thinking},"
    payload+="\"use_format\":${use_format},"
    payload+="\"use_cot_caption\":${def_cot_caption:-true},"
    payload+="\"use_cot_language\":${def_cot_language:-true},"
    payload+="\"vocal_language\":\"${language}\","
    payload+="\"audio_format\":\"${def_audio_format:-mp3}\","
    payload+="\"use_random_seed\":true"

    [ -n "$model" ] && payload+=",\"model\":\"$(json_escape "$model")\""
    [ -n "$steps" ] && payload+=",\"inference_steps\":${steps}"
    [ -n "$guidance" ] && payload+=",\"guidance_scale\":${guidance}"
    [ -n "$seed" ] && payload+=",\"seed\":${seed},\"use_random_seed\":false"
    [ -n "$duration" ] && payload+=",\"audio_duration\":${duration}"
    [ -n "$bpm" ] && payload+=",\"bpm\":${bpm}"
    [ -n "$batch" ] && payload+=",\"batch_size\":${batch}"

    payload+="}"

    echo "Generating music..."
    if [ -n "$description" ]; then
        echo "  Mode: Simple (description)"
        echo "  Description: ${description:0:50}..."
    else
        echo "  Mode: Caption"
        echo "  Caption: ${caption:0:50}..."
    fi
    echo "  Thinking: $thinking, Format: $use_format"
    echo "  Output: $OUTPUT_DIR"
    echo ""

    # Write payload to temp file to ensure UTF-8 encoding
    local temp_payload=$(mktemp)
    printf "%s" "$payload" > "$temp_payload"

    local response=$(curl -s -X POST "${api_url}/v1/music/generate" \
        -H "Content-Type: application/json; charset=utf-8" \
        --data-binary "@${temp_payload}")

    rm -f "$temp_payload"

    local job_id=$(json_get "$response" "job_id")

    [ -z "$job_id" ] && { echo -e "${RED}Error: Failed to create job${NC}"; echo "$response"; exit 1; }

    if [ "$no_wait" = true ]; then
        echo "Job ID: $job_id"
        echo "Use '$0 status $job_id' to check progress and download"
    else
        wait_for_job "$api_url" "$job_id"
    fi
}

# Random command
cmd_random() {
    check_deps
    ensure_config

    local thinking="" no_thinking=false no_wait=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            --thinking|-t) thinking="true"; shift ;;
            --no-thinking) no_thinking=true; shift ;;
            --no-wait) no_wait=true; shift ;;
            *) shift ;;
        esac
    done

    local api_url=$(ensure_connection)

    local def_thinking=$(get_config "generation.thinking")
    [ -z "$thinking" ] && thinking="${def_thinking:-true}"
    [ "$no_thinking" = true ] && thinking="false"

    echo "Generating random music..."
    echo "  Thinking: $thinking"
    echo "  Output: $OUTPUT_DIR"
    echo ""

    local temp_payload=$(mktemp)
    printf "{\"thinking\": %s}" "$thinking" > "$temp_payload"

    local response=$(curl -s -X POST "${api_url}/v1/music/random" \
        -H "Content-Type: application/json; charset=utf-8" \
        --data-binary "@${temp_payload}")

    rm -f "$temp_payload"

    local job_id=$(json_get "$response" "job_id")

    [ -z "$job_id" ] && { echo -e "${RED}Error: Failed to create job${NC}"; echo "$response"; exit 1; }

    if [ "$no_wait" = true ]; then
        echo "Job ID: $job_id"
        echo "Use '$0 status $job_id' to check progress and download"
    else
        wait_for_job "$api_url" "$job_id"
    fi
}

# Help
show_help() {
    echo "ACE-Step Music Generation CLI"
    echo ""
    echo "Requirements: curl"
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  generate    Generate music from text"
    echo "  random      Generate random music"
    echo "  status      Check job status and download results"
    echo "  models      List available models"
    echo "  health      Check API health"
    echo "  config      Manage configuration"
    echo ""
    echo "Output:"
    echo "  Results saved to: $OUTPUT_DIR/<job_id>.json"
    echo "  Audio files: $OUTPUT_DIR/<job_id>_1.mp3, ..."
    echo ""
    echo "Generate Options:"
    echo "  -c, --caption     Music style/genre description (caption mode)"
    echo "  -d, --description Simple description, LM auto-generates caption/lyrics"
    echo "  -l, --lyrics      Lyrics text"
    echo "  -t, --thinking    Enable thinking mode (default: true)"
    echo "  --no-thinking     Disable thinking mode"
    echo "  --no-format       Disable format enhancement"
    echo ""
    echo "Examples:"
    echo "  $0 generate \"Pop music with guitar\"           # Caption mode"
    echo "  $0 generate -d \"A February love song\"         # Simple mode (LM generates)"
    echo "  $0 generate -c \"Jazz\" -l \"[Verse] Hello\"      # With lyrics"
    echo "  $0 random"
    echo "  $0 status <job_id>"
    echo "  $0 config --set generation.thinking false"
}

# Main
case "$1" in
    generate) shift; cmd_generate "$@" ;;
    random) shift; cmd_random "$@" ;;
    status) shift; cmd_status "$@" ;;
    models) cmd_models ;;
    health) cmd_health ;;
    config) shift; cmd_config "$@" ;;
    help|--help|-h) show_help ;;
    *) show_help; exit 1 ;;
esac
