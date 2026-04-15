from datetime import datetime
import json
from Schemes.schema import OverallState


def Manual_process_ocr_page(state: OverallState):

    with open('files_log/final_ocr_results.json', 'r') as f:
        ocr_results = json.load(f)
        
    return {"ocr_results": ocr_results}