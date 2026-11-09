from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sqlite3
import time
import uuid
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mock_llm
ROOT=Path(__file__).resolve().parents[1]

POL=ROOT/"data"/"policies"
class PolicyRAG:
    def __init__(self,chunk_size=40,overlap=10):
        self.chunk_size=chunk_size; self.overlap=overlap; self.chunks=[]
        for path in sorted(POL.glob("*.txt")):
            words=path.read_text(encoding="utf-8").split()
            step=max(1,chunk_size-overlap)
            for start in range(0,len(words),step):
                chunk=words[start:start+chunk_size]
                self.chunks.append({"source":path.name,"text":" ".join(chunk)})
                if start+chunk_size>=len(words): break
        self.vectorizer=TfidfVectorizer()
        self.matrix=self.vectorizer.fit_transform([x["text"] for x in self.chunks])
    def retrieve(self,query,k=3):
        q=self.vectorizer.transform([query]); scores=cosine_similarity(q,self.matrix)[0]
        idx=scores.argsort()[::-1][:k]
        return [{"source":self.chunks[i]["source"],"score":float(scores[i]),"text":self.chunks[i]["text"]} for i in idx]
    def answer(self,question):
        hits=self.retrieve(question,3)
        context="\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
        response=mock_llm.chat([{"role":"system","content":"POLICY ASSISTANT: answer only from supplied context. If not present, say I could not find that information in the policy documents."},
          {"role":"user","content":f"CONTEXT:\n{context}\nQUESTION:\n{question}"}])
        return {"answer":response.choices[0].message.content,"sources":list(dict.fromkeys(h["source"] for h in hits)),"retrieved":hits}
def evaluate(rag):
    df=pd.read_csv(ROOT/"data/raw/policy_eval.csv",keep_default_na=False); out=[]
    for _,r in df.iterrows():
        ans=rag.answer(r.question); expected_doc=r.expected_doc; fact=r.expected_fact
        out.append({"question":r.question,"expected_doc":expected_doc,"retrieved_doc":expected_doc in ans["sources"] if expected_doc!="NONE" else False,
          "fact_found":fact in ans["answer"] if fact else False,"fallback":ans["answer"].startswith("I could not find")})
    return pd.DataFrame(out)
DB=ROOT/"data/instamart.db"; OUT=ROOT/"outputs"; STATE=ROOT/"state"
def get_order(order_id):
    with sqlite3.connect(DB) as con:
        row=con.execute("select order_id,status,order_amount,delivery_minutes from orders where order_id=?",(order_id,)).fetchone()
    return {"error":"order not found"} if not row else dict(zip(["order_id","status","order_amount","delivery_minutes"],row))
def search_policy(query): return {"results":rag.retrieve(query,3)}
def calculate_refund(order_amount,delivery_minutes):
    amount=float(order_amount) if delivery_minutes is not None and float(delivery_minutes)>45 else 0.0
    return {"refund_inr":round(amount,2)}
def _ledger(tool,order_id,refund_inr):
    OUT.mkdir(exist_ok=True); path=OUT/"refund_ledger.jsonl"
    with path.open("a",encoding="utf-8") as f: f.write(json.dumps({"timestamp":datetime.now(timezone.utc).isoformat(),"tool":tool,"order_id":order_id,"refund_inr":refund_inr})+"\n")
    return {"refund_inr":refund_inr}
def issue_refund(order_id,refund_inr): return _ledger("issue_refund",order_id,refund_inr)
def escalate_to_human(order_id,refund_inr): return _ledger("escalate_to_human",order_id,refund_inr)
TOOL_MAP={"get_order":get_order,"search_policy":search_policy,"calculate_refund":calculate_refund,"issue_refund":issue_refund,"escalate_to_human":escalate_to_human}
TOOLS_SPEC=[{"name":n,"description":f"Execute the {n} refund workflow operation.","parameters":{"type":"object","properties":{}}} for n in TOOL_MAP]
def run_agent(user_message,scenario_id="demo",max_iterations=6,token_limit=6000,time_limit_seconds=10,run_id=None,crash_after=None):
    run_id=run_id or str(uuid.uuid4()); STATE.mkdir(exist_ok=True)
    messages=[{"role":"system","content":"AGENT: Use tools to verify orders and follow refund policy."},{"role":"user","content":user_message}]
    tokens=0; steps=[]; executed=[]; start=time.monotonic(); llm_calls=0
    def checkpoint(step):
        obj={"run_id":run_id,"scenario_id":scenario_id,"messages":messages,"tokens_used":tokens,"step_number":step,"tools_executed":executed,"steps":steps,"llm_calls":llm_calls}
        (STATE/f"{run_id}.json").write_text(json.dumps(obj,default=str),encoding="utf-8")
    for iteration in range(max_iterations):
        if time.monotonic()-start>=time_limit_seconds:
            return {"status":"FALLBACK_TIMEOUT","final_answer":"FALLBACK_TIMEOUT: hand off to a human agent.","steps":steps,"llm_calls":llm_calls,"tokens_used":tokens}
        response=mock_llm.chat(messages); llm_calls+=1; tokens+=response.usage.total_tokens
        if tokens>token_limit:
            return {"status":"FALLBACK_TOKEN_LIMIT","final_answer":"FALLBACK_TOKEN_LIMIT: hand off to a human agent.","steps":steps,"llm_calls":llm_calls,"tokens_used":tokens}
        msg=response.choices[0].message
        if not msg.tool_calls:
            content=msg.content or ""; return {"status":content.split(":",1)[0],"final_answer":content,"steps":steps,"llm_calls":llm_calls,"tokens_used":tokens}
        tc=msg.tool_calls[0]; args=tc.arguments
        messages.append({"role":"assistant","content":None,"tool_calls":[{"id":tc.id,"name":tc.name,"arguments":args}]})
        observation=TOOL_MAP[tc.name](**args); executed.append(tc.name)
        steps.append({"tool":tc.name,"arguments":args,"observation":observation})
        messages.append({"role":"tool","tool_call_id":tc.id,"name":tc.name,"content":json.dumps(observation)})
        checkpoint(iteration+1)
        if crash_after is not None and len(executed)==crash_after: raise RuntimeError("Simulated crash after checkpoint")
    return {"status":"FALLBACK_MAX_ITERATIONS","final_answer":"FALLBACK_MAX_ITERATIONS: hand off to a human agent.","steps":steps,"llm_calls":llm_calls,"tokens_used":tokens}
