import base64
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

VISION_MODEL = "gpt-5.1"

VISION_PROMPT = """
Analyze this image for a Software QA Engineer. Provide a structured breakdown of the interface to aid in writing automated test cases.

Output the description in the following sections:

1. Screen Overview
2. Interactive Elements (Buttons & Inputs)
3. Static Text & Instructions
4. UI State & Feedback
5. Visual Layout

Ensure all extracted text is accurate for test-case validation.
"""

def process_image(image_path):
    try:
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")

        llm = ChatOpenAI(model=VISION_MODEL, max_tokens=3000)

        message = HumanMessage(content=[
            {"type": "text", "text": VISION_PROMPT},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}
            }
        ])

        response = llm.invoke([message])

        return Document(
            page_content=response.content.strip(),
            metadata={
                "source": image_path,
                "doc_type": "image",
                "type": "image"
            }
        )

    except Exception as e:
        print(f"[Image] Failed to process {image_path}: {e}")
        return None