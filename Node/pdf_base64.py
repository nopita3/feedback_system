from io import BytesIO
import fitz  # PyMuPDF
import pandas as pd
import base64


def read_and_split_pdf(state):
    if "student_test_path" in state and isinstance(state["student_test_path"], bytes):
        doc = fitz.open(stream=state["pdf_path"], filetype="pdf")
        df = pd.read_csv(BytesIO(state["student_test_path"])).sample(5, random_state=42)

        # Convert numpy types to native Python types using str()
        key_list = [ {col: str(df.loc[0, col])} for col in df.columns.to_list() if col.startswith("PriKey") and col[-1].isdigit() ][:25]
        pages_list = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            zoom = 1.25
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            
            img_bytes = pix.tobytes("png")
            b64_img = base64.b64encode(img_bytes).decode("utf-8")
            pages_list.append(b64_img)
            
        return {"pages": pages_list , "key_answer": key_list}
    
    elif isinstance(state["curriculum_path"],str):
        doc = fitz.open(state['curriculum_path'])
        pages_list = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            zoom = 1.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            
            img_bytes = pix.tobytes("png")
            b64_img = base64.b64encode(img_bytes).decode("utf-8")
            pages_list.append(b64_img)
            
        return {"pages": pages_list}
    else:
        raise ValueError("Invalid state: missing 'student_test_path' or 'curriculum_path'")
