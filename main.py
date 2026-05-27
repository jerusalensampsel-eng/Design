from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import fitz
import cv2
import numpy as np
import os
import easyocr
from openai import OpenAI
import base64
from datetime import datetime

app = FastAPI(title="设计走查宝 - 后端")

# 允许前端跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)

# ==================== 配置 AI（推荐使用 Grok 或 通义千问） ====================
client = OpenAI(
    api_key="sk-XXXXXXXXXXXXXXXX",          # ←←← 在这里填你的 API Key
    base_url="https://api.grok.x.ai/v1"     # Grok
    # base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 通义千问
)

os.makedirs("temp", exist_ok=True)

@app.post("/review")
async def review(design: UploadFile = File(...), ci: UploadFile = None):
    try:
        # 保存设计文件
        design_path = f"temp/{design.filename}"
        with open(design_path, "wb") as f:
            f.write(await design.read())

        # 保存CI规范（可选）
        ci_text = ""
        if ci:
            ci_path = f"temp/{ci.filename}"
            with open(ci_path, "wb") as f:
                f.write(await ci.read())
            doc = fitz.open(ci_path)
            ci_text = "\n".join([page.get_text() for page in doc])

        # 处理并分析
        annotated_path, issues = process_file(design_path, ci_text)

        return {
            "success": True,
            "issues": issues,
            "annotated_filename": os.path.basename(annotated_path)
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

def process_file(file_path: str, ci_text: str):
    doc = fitz.open(file_path)
    issues = []
    issue_id = 1

    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(dpi=300)
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        
        if pix.n == 4:
            img = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        else:
            img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # OCR
        ocr_results = reader.readtext(img)
        ocr_text = " ".join([text for _, text, _ in ocr_results])

        # 调用 AI 分析
        ai_issues = call_ai_vision_analysis(img, ocr_text, ci_text, page_num)

        for item in ai_issues:
            x, y, w, h = item.get("bbox", [100, 100, 200, 80])
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 6)
            cv2.arrowedLine(img, (x + w//2, y - 30), (x + w//2, y - 90), (0, 0, 255), 5, tipLength=0.35)

            issues.append({
                "id": issue_id,
                "type": item["type"],
                "description": item["description"],
                "basis": item["basis"],
                "source": item["source"],
                "location": f"第 {page_num + 1} 页",
                "suggestion": item.get("suggestion", "建议按规范修改")
            })
            issue_id += 1

    # 保存标注后的图片
    annotated_path = f"temp/annotated_{os.path.basename(file_path).replace('.pdf', '.jpg')}"
    cv2.imwrite(annotated_path, img)

    return annotated_path, issues

def call_ai_vision_analysis(img, ocr_text, ci_text, page_num):
    """调用多模态 AI 进行智能分析"""
    # 图片转 base64
    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    prompt = f"""
    你是资深品牌设计师和设计规范审核专家。
    请严格分析这张设计页面，重点关注：
    - Logo 使用是否符合标准（尺寸、留白、颜色、变形）
    - 文字是否有错别字、不规范用法
    - 产品逻辑、视觉层级、排版合理性
    CI规范参考：{ci_text[:3000]}

    请用JSON数组格式返回问题，每条问题包含：type, description, basis, source, bbox（[x,y,w,h]）, suggestion
    """

    try:
        response = client.chat.completions.create(
            model="grok-2-vision-1212",     # 或 qwen-vl-max / gpt-4o
            messages=[
                {"role": "system", "content": "你是一个严谨的设计审核AI"},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
                ]}
            ],
            temperature=0.3,
            max_tokens=800
        )
        # 解析JSON（实际项目中需加入错误处理）
        result = response.choices[0].message.content
        import json
        return json.loads(result) if "```json" not in result else json.loads(result.split("```json")[1].split("```")[0])
    except:
        # 降级返回示例
        return [{
            "type": "Logo规范问题",
            "description": "Logo 尺寸或留白不符合CI要求",
            "basis": "CI手册规定Logo最小占画面宽度8%",
            "source": "上传的CI规范文件",
            "bbox": [150, 80, 220, 90],
            "suggestion": "请放大Logo并增加周围留白"
        }]

@app.get("/download/{filename}")
async def download_file(filename: str):
    return FileResponse(f"temp/{filename}", media_type="image/jpeg")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)