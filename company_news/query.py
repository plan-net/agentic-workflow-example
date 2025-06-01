import datetime

from jinja2 import Template
from openai import OpenAI
import ray

@ray.remote
def query(text: str, 
          start: datetime.datetime, 
          end: datetime.datetime) -> dict:
    template = Template("""
    Identify news from {{start}} to {{end}} about company **"{{name}}"**. 
    Format the output as a bullet point list in the following format:

    * YYYY-mm-dd - [**Headline**](Link): Brief Summary of the news.
                        
    Only output the bullet point list about news in the specified date range. 
    Do not include any other text or additional information. If you cannot find 
    any news for the given date range then output the text "no news found".    
    """)
    try:
        resp = chat(
            template.render(
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                name=text)
        )
        return {**resp, **{
            "query": text,
            "start": start,
            "end": end,
            "error": None,
        }}
    except Exception as e:
        return {
            "query": text,
            "start": start,
            "end": end,
            "error": e
        }

def chat(query: str, model="gpt-4o-mini"):
    t0 = datetime.datetime.now()
    client = OpenAI()
    response = client.responses.create(
        model=model,
        tools=[{"type": "web_search_preview"}],
        input=query
    )
    runtime = datetime.datetime.now() - t0
    return {
        "response": response.model_dump(),
        "output": response.output_text,
        "model": model,
        "runtime": runtime.total_seconds()
    }

from typing import List

# def batch(texts: List[str],
#           start: datetime.datetime,
#           end: datetime.datetime):

#     refined = [t.strip() for t in texts if t.strip()]
#     futures = [query.remote(t, start, end) for t in refined]
#     remaining_futures = futures.copy()
#     completed_jobs = 0
#     results = []

#     while remaining_futures:
#         done_futures, remaining_futures = ray.wait(
#             remaining_futures, num_returns=1)
#         result = ray.get(done_futures[0])
#         results.append(result)
#         completed_jobs += 1
#         p = completed_jobs / len(texts) * 100.
#         print(f"{result['query']}\n{completed_jobs}/{len(texts)} = {p:.0f}%")
#         if result["error"]:
#             print(f"**Error:** {result['error']}")
#         else:
#             print(result["output"])

#     return results


from kodosumi.core import ServeAPI
app = ServeAPI()


from kodosumi.core import forms as F

news_model = F.Model(
    F.Markdown("""
    # Search News
    Specify the _query_ - for example the name of your client, the start and end date. You can specify multiple query. Type one query per line.
    """),
    F.Break(),
    F.InputArea(label="Query", name="texts"),
    F.InputDate(label="Start Date", name="start", required=True),
    F.InputDate(label="End Date", name="end", required=True),
    F.Submit("Submit"),
    F.Cancel("Cancel")
)


import fastapi
from kodosumi.core import InputsError
from kodosumi.core import Launch
from kodosumi.core import Tracer
import asyncio
from typing import Optional, List


@app.enter(
    path="/",
    model=news_model,
    summary="News Search",
    description="Search for news.",
    tags=["OpenAI"],
    version="1.0.0",
    author="m.rau@house-of-communication.com")
async def enter(request: fastapi.Request, inputs: dict):
    # parse and cleanse inputs
    query = inputs.get("texts", "").strip()
    start = datetime.datetime.strptime(inputs.get("start"), "%Y-%m-%d")
    end = datetime.datetime.strptime(inputs.get("end"), "%Y-%m-%d")
    texts = [s.strip() for s in query.split("\n") if s.strip()]
    # validate inputs
    error = InputsError()
    if not texts:
        error.add(texts="Please specify a query to search for news.")
    if start > end:
        error.add(start="Must be before or equal to end date.")
    if error.has_errors():
        raise error
    # launch execution
    return Launch(
        request, 
        "company_news.query:run_batch", 
        inputs={"texts": texts, "start": start, "end": end}
    )

from ray import serve

@serve.deployment
@serve.ingress(app)
class NewsSearch: pass

fast_app = NewsSearch.bind()


async def run_batch(inputs: dict, tracer: Tracer):
    texts = inputs.get("texts", [])
    start = inputs.get("start", datetime.datetime.now())
    end = inputs.get("end", datetime.datetime.now())
    return await batch(texts, start, end, tracer)


async def batch(texts: List[str], 
                start: datetime.datetime, 
                end: datetime.datetime,
                tracer: Optional[Tracer]=None):
    refined = [t.strip() for t in texts if t.strip()]
    futures = [query.remote(t, start, end) for t in refined]
    unready = futures.copy()
    completed_jobs = 0
    results = []
    while unready:
        ready, unready = ray.wait(unready, num_returns=1, timeout=1)
        if ready:
            result = ray.get(ready[0])
            results.append(result)
            completed_jobs += 1
            p = completed_jobs / len(texts) * 100.
            await tracer.markdown(f"#### {result['query']}")
            await tracer.markdown(f"{completed_jobs}/{len(texts)} = {p:.0f}%")
            if result["error"]:
                await tracer.markdown(f"**Error:** {result['error']}")
            else:
                await tracer.markdown(result["output"])
            await tracer.html("<div class='large-divider'></div>")
            print(f"Job completed ({completed_jobs}/{len(texts)})")
        await asyncio.sleep(1)
    return results
