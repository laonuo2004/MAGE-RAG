import PyPDF2
import pytesseract
from pdf2image import convert_from_path
import traceback
import logging
from io import BytesIO
import base64
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
import json
import re
from config.config import LLM_BASE_URL, LLM_API_KEY
from config.config import PROMPTS
logger = logging.getLogger(__name__)
from loguru import logger as loguru_logger
import aiohttp

async def get_pdf(url: str, filename: str):
    i = 0
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                with open(filename, 'wb') as f:
                    while True:
                        chunk = await response.content.read(81920)
                        i += 81920
                        if not chunk:
                            break
                        f.write(chunk)
            else:
                raise Exception("Failed to download pdf")

def clean_text(text):
    """
    Clean control characters from text to prevent JSON parsing errors
    
    Args:
        text (str): Original text
        
    Returns:
        str: Cleaned text
    """
    if not text:
        return text
    
    control_chars = ''.join(map(chr, range(0, 32)))
    control_chars = control_chars.replace('\t', '').replace('\n', '').replace('\r', '')
    
    control_char_re = re.compile('[%s]' % re.escape(control_chars))
    cleaned = control_char_re.sub(' ', text)
    
    cleaned = cleaned.replace('\x00', '')
    
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    return cleaned.strip()

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file using OCR if needed.
    Clean control characters to prevent JSON parsing errors.
    Args:
        pdf_path (str): Path to the PDF file
    Returns:
        list: List of cleaned text from each page
    """
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file, strict=False)
            pages = [page.extract_text() for page in reader.pages]
            
        if any(not page.strip() for page in pages):
            logger.info(f"Using OCR for {pdf_path} as some pages have no text")
            pages = []
            pdf_images = convert_from_path(pdf_path)
            for page_num, page_img in enumerate(pdf_images):
                text = pytesseract.image_to_string(page_img)
                pages.append(f"--- Page {page_num + 1} ---\n{text}\n")
        
        cleaned_pages = [clean_text(page) for page in pages]
        
        return cleaned_pages
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
        traceback.print_exc()
        return []

text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)

def split_text(text):
    """
    Split text into chunks.
    
    Args:
        text (str): Text to split
        
    Returns:
        list: List of text chunks
    """
    return text_splitter.split_text(text)


def extract_images_from_pdf(pdf_path):
    """
    Extract images from a PDF file.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        list: List of images
    """
    return convert_from_path(pdf_path, dpi=100)


def encode_image(pil_image):
    """Encode a PIL image to base64 string."""
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")    
    return img_str

client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

def analyze_content(content, visual, model="gpt-4o"):
    """Analyze content to extract keywords, context, and other metadata"""

    response_format={"type": "json_schema", "json_schema": {
                        "name": "response",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "keywords": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "context": {
                                    "type": "string",
                                },
                                "tags": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                            },
                            "required": ["keywords", "context", "tags"],
                            "additionalProperties": False
                        },
                        "strict": True
                }
            }

    if visual==False:
        content = clean_text(content)
        
        prompt = PROMPTS["text"] + content

        try:
            response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You must respond with a JSON object."},
                {"role": "user", "content": prompt}
            ],
            response_format=response_format,
            temperature=0.7,
            max_tokens=2048)
            
            analysis = json.loads(response.choices[0].message.content)

            return analysis
            
        except Exception as e:
            error_msg = str(e)
            loguru_logger.error(f"Text content analysis failed: {error_msg}")
            loguru_logger.info(f"Content length: {len(content)} characters")
            if "Invalid control character" in error_msg or "control character" in error_msg.lower():
                loguru_logger.warning("Control character issue detected!")
                control_chars_found = [char for char in content if ord(char) < 32 and char not in '\t\n\r']
                if control_chars_found:
                    loguru_logger.info(f"Control characters found: {[hex(ord(c)) for c in control_chars_found[:10]]}")
                loguru_logger.info("Note: Text has been automatically cleaned, if error persists check API response")
            loguru_logger.debug(f"Content preview: {content[:200]}...")
            loguru_logger.exception("Text content analysis exception")
            return {
                "keywords": ["analysis_failed"],
                "summary": "LLM analysis failed, unable to extract summary",
                "category": "error",
                "tags": ["analysis_failed", "exception"]
            }

    else:
        prompt = PROMPTS["image"]

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You must respond with a JSON object."},
                    {"role": "user", "content": [
                        {"type":"text","text": prompt},
                        {"type": "image_url","image_url": {"url": f"data:image/jpeg;base64,{content}"}}
                    ]},
                ],
                response_format=response_format,
                temperature=0.7,
                max_tokens=2048
            )
            analysis = json.loads(response.choices[0].message.content)

            return analysis
            
        except Exception as e:
            loguru_logger.error(f"Image content analysis failed: {str(e)}")
            loguru_logger.info(f"Image data length: {len(content)} characters (base64 encoded)")
            loguru_logger.exception("Image content analysis exception")
            return {
                "keywords": ["analysis_failed"],
                "summary": "LLM image analysis failed, unable to extract image description",
                "category": "error",
                "tags": ["analysis_failed", "exception", "image"]
            }
