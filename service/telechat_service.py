import uvicorn
import os

from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from fastapi.encoders import jsonable_encoder
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
PATH = '../models/7B'
tokenizer = AutoTokenizer.from_pretrained(PATH)
model = AutoModelForCausalLM.from_pretrained(PATH, trust_remote_code=True, device_map="auto",
                                             torch_dtype=torch.float16)
generate_config = GenerationConfig.from_pretrained(PATH)
model.eval()
print("=============AIGC服务启动==========")


def _gc():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    _gc()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory for serving frontend assets
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


def check_ex(do_sample, max_length, top_k, top_p, temperature, repetition_penalty):
    flag = True
    try:
        if do_sample != None and (do_sample not in [True, False] or not isinstance(do_sample, bool)):
            flag = False
        if max_length != None and (not 0 < max_length <= 4096 or not isinstance(max_length, int)):
            flag = False
        if top_k != None and (not 0 < top_k < 100 or not isinstance(top_k, int)):
            flag = False
        if top_p != None and (not 0.0 < top_p < 1.0 or not isinstance(top_p, float)):
            flag = False
        if temperature != None and (not 0.0 < temperature < 1.0 or not isinstance(temperature, float)):
            flag = False
        if repetition_penalty != None and (
                not 1.0 < repetition_penalty < 100.0 or not isinstance(repetition_penalty, float)):
            flag = False
        return flag
    except Exception:
        flag = False
        return flag


def streamresponse_v2(tokenizer, query, history, do_sample, max_length, top_k, top_p, temperature, repetition_penalty):
    result_generator = model.chat(tokenizer, query, history=history, generation_config=generate_config, stream=True,
                                  do_sample=do_sample, max_length=max_length, top_k=top_k, temperature=temperature,
                                  repetition_penalty=repetition_penalty, top_p=top_p)
    t_resp = ''
    while 1:
        try:
            char, _ = next(result_generator)
            if char is None:
                break
            else:
                t_resp += char

                yield char
        except StopIteration:
            break


def response_data(seqid, code, message, flag, data):
    res_dict = {
        "seqid": seqid,
        "code": code,
        "message": message,
        "flag": flag,
        "data": data
    }
    res = jsonable_encoder(res_dict)
    print("### 整个接口的返回结果: ", res)
    return res


def parse_data(dialog):
    history = dialog[:-1]
    query = dialog[-1].get("content")
    return history, query


@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint serving HTML page with Vercel Web Analytics integration."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>TeleChat AIGC Service</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 20px;
            }
            .container {
                background: white;
                border-radius: 12px;
                box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
                max-width: 800px;
                width: 100%;
                padding: 40px;
            }
            h1 {
                color: #333;
                margin-bottom: 20px;
                text-align: center;
            }
            .info {
                background: #f5f5f5;
                border-left: 4px solid #667eea;
                padding: 15px;
                margin-bottom: 20px;
                border-radius: 4px;
            }
            .endpoint {
                background: #f9f9f9;
                border: 1px solid #e0e0e0;
                padding: 15px;
                margin-bottom: 15px;
                border-radius: 4px;
                font-family: 'Courier New', monospace;
                font-size: 13px;
            }
            .endpoint-method {
                color: #667eea;
                font-weight: bold;
            }
            .endpoint-path {
                color: #764ba2;
            }
            .feature-list {
                list-style: none;
                padding-left: 0;
            }
            .feature-list li {
                padding: 8px 0;
                color: #555;
            }
            .feature-list li:before {
                content: "✓ ";
                color: #667eea;
                font-weight: bold;
                margin-right: 8px;
            }
            .footer {
                text-align: center;
                margin-top: 30px;
                color: #999;
                font-size: 12px;
            }
            .analytics-badge {
                text-align: center;
                margin-top: 20px;
                padding-top: 20px;
                border-top: 1px solid #e0e0e0;
                color: #999;
                font-size: 11px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 TeleChat AIGC Service</h1>
            
            <div class="info">
                <strong>Welcome!</strong> This FastAPI service provides TeleChat AI model capabilities with Vercel Web Analytics integration.
            </div>

            <h2 style="font-size: 18px; margin-top: 30px; margin-bottom: 15px; color: #333;">Available Endpoints</h2>
            
            <div class="endpoint">
                <span class="endpoint-method">POST</span> <span class="endpoint-path">/telechat/gptDialog/v2</span>
                <p style="margin-top: 8px; color: #666;">Streaming response endpoint for chat dialog</p>
            </div>

            <div class="endpoint">
                <span class="endpoint-method">POST</span> <span class="endpoint-path">/telechat/gptDialog/v4</span>
                <p style="margin-top: 8px; color: #666;">Non-streaming response endpoint for chat dialog</p>
            </div>

            <h2 style="font-size: 18px; margin-top: 30px; margin-bottom: 15px; color: #333;">Features</h2>
            <ul class="feature-list">
                <li>Multi-turn conversation support</li>
                <li>Configurable generation parameters</li>
                <li>Stream and non-stream modes</li>
                <li>CORS enabled for cross-origin requests</li>
                <li>Vercel Web Analytics integrated</li>
            </ul>

            <div class="analytics-badge">
                📊 Powered by Vercel Web Analytics<br/>
                Tracking user interactions and page performance
            </div>

            <div class="footer">
                TeleChat © 2024 | FastAPI Service
            </div>
        </div>

        <!-- Vercel Web Analytics Script -->
        <script>
            // Vercel Web Analytics Injection
            // This script enables automatic tracking of user interactions and page performance
            // Replace 'your-vercel-project-id' with your actual Vercel project ID
            
            (function() {
                // Detect if already loaded
                if (window.__VERCEL_ANALYTICS) {
                    return;
                }
                
                window.__VERCEL_ANALYTICS = {
                    version: '0.1.0'
                };
                
                // Analytics tracking function
                function track(type, data) {
                    try {
                        // Log to console for debugging
                        console.log('[Vercel Analytics]', type, data);
                        
                        // Send beacon to Vercel analytics endpoint
                        // This will be handled by Vercel when deployed
                        if (navigator.sendBeacon) {
                            const payload = JSON.stringify({
                                event: type,
                                timestamp: Date.now(),
                                url: window.location.href,
                                ...data
                            });
                            navigator.sendBeacon('/_vercel/insights/view', payload);
                        }
                    } catch (e) {
                        console.error('[Vercel Analytics Error]', e);
                    }
                }
                
                // Track page view
                track('pageview', {
                    page: window.location.pathname
                });
                
                // Track clicks
                document.addEventListener('click', function(e) {
                    if (e.target.tagName === 'A' || e.target.tagName === 'BUTTON') {
                        track('interaction', {
                            type: 'click',
                            element: e.target.tagName,
                            text: e.target.textContent
                        });
                    }
                });
                
                // Track page visibility changes
                document.addEventListener('visibilitychange', function() {
                    track('visibility', {
                        visible: !document.hidden
                    });
                });
            })();
        </script>
    </body>
    </html>
    """
    return html_content


@app.post('/telechat/gptDialog/v2')
async def doc_gptDialog_v2(item: dict):
    session_res = []
    # 参数校验
    try:
        dialog = item["dialog"]
    except:
        result_info = response_data("", "10301", "服务必填参数缺失", "0", "执行失败")
        return result_info
    # 暴露参数读取
    do_sample = item.get("do_sample", True)
    max_length = item.get("max_length", 4096)
    top_k = item.get("top_k", 20)
    top_p = item.get("top_p", 0.2)
    temperature = item.get("temperature", 0.1)
    repetition_penalty = item.get("repetition_penalty", 1.03)
    odd = [i for i in range(len(dialog)) if i % 2 == 0]
    even = [x for x in range(len(dialog)) if x % 2 == 1]
    for index in odd:
        if dialog[index].get("role", "") != "user":
            result_info = response_data("", "10904", "服务请求参数dialog错误", "0", "执行失败")
            return result_info
    for index in even:
        if dialog[index].get("role", "") != "bot":
            result_info = response_data("", "10904", "服务请求参数dialog错误", "0", "执行失败")
            return result_info
    if not check_ex(do_sample, max_length, top_k, top_p, temperature, repetition_penalty):
        result_info = response_data("", "10305", "请求参数范围错误", "0", "执行失败")
        return result_info
    try:
        history, query = parse_data(dialog)
        return StreamingResponse(
            streamresponse_v2(tokenizer, query, history, do_sample, max_length, top_k, top_p, temperature,
                              repetition_penalty),
            media_type="text/html")
    except Exception:
        import traceback
        traceback.print_exc()
        result_info = response_data('', "10903", "服务执行失败", "0", "执行失败")
        return result_info


@app.post('/telechat/gptDialog/v4')
async def doc_gptDialog_v3(item: dict, ):
    session_res = []
    try:
        dialog = item["dialog"]
    except:
        result_info = response_data("", "10301", "服务必填参数缺失", "0", "执行失败")
        return result_info
    odd = [i for i in range(len(dialog)) if i % 2 == 0]
    even = [x for x in range(len(dialog)) if x % 2 == 1]
    for index in odd:
        if dialog[index].get("role", "") != "user":
            result_info = response_data("", "10904", "服务请求参数dialog错误", "0", "执行失败")
            return result_info
    for index in even:
        if dialog[index].get("role", "") != "bot":
            result_info = response_data("", "10904", "服务请求参数dialog错误", "0", "执行失败")
            return result_info
    do_sample = item.get("do_sample", True)
    max_length = item.get("max_length", 4096)
    top_k = item.get("top_k", 20)
    top_p = item.get("top_p", 0.2)
    temperature = item.get("temperature", 0.1)
    repetition_penalty = item.get("repetition_penalty", 1.03)
    if not check_ex(do_sample, max_length, top_k, top_p, temperature, repetition_penalty):
        result_info = response_data("", "10305", "请求参数范围错误", "0", "执行失败")
        return result_info
    try:
        history, query = parse_data(dialog)
        t_resp = model.chat(tokenizer, query, history=history, generation_config=generate_config, stream=False,
                            do_sample=do_sample, max_length=max_length, top_k=top_k, temperature=temperature,
                            repetition_penalty=repetition_penalty, top_p=top_p)

        res_data = {
            'role': "bot",
            'content': t_resp
        }
        result_info = res_data
    except Exception:
        import traceback
        traceback.print_exc()
        result_info = response_data('', "10903", "服务执行失败", "0", "执行失败")
    return result_info


if __name__ == "__main__":
    ip = "0.0.0.0"
    port = 8070
    uvicorn.run(app, host=ip, port=port, reload=False)
