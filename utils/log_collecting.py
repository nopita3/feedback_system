import json
from langchain_core.callbacks import UsageMetadataCallbackHandler
from fcntl import flock, LOCK_EX, LOCK_UN
from pathlib import Path

base_dir = Path(__file__).cwd()
log_file_path = base_dir / "Token_usage_log.txt"

def log_token_usage(callback: UsageMetadataCallbackHandler, 
                    start_date, 
                    processtime,
                    platform,
                    agent_work
                   ):
    
    token_metadata = {
                "connection_time": str(start_date),
                "token_usage" : callback.usage_metadata,
                "processing_time": processtime,
                "agent_work": agent_work,
                'platform': platform
            }

    try:
        with open(f"{log_file_path}", "a", encoding="utf-8") as f:
            flock(f.fileno(), LOCK_EX)
            f.write(json.dumps(token_metadata, ensure_ascii=False, default=str) + "\n")
            flock(f.fileno(), LOCK_UN)
            
    except Exception as e:
        print(f"Error logging token usage: {e}")
                