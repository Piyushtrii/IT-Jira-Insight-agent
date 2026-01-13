import os
import streamlit as st
import pandas as pd
import faiss
import numpy as np
import plotly.express as px
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from typing import TypedDict, List, Annotated, Dict, Any
from operator import add
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")


@st.cache_resource
def get_llm():
    """Load LLM securely with Streamlit Secrets (Cloud) + .env fallback (Local)"""
    try:
        groq_api_key = st.secrets.get("GROQ_API_KEY", "")
        
        if not groq_api_key:
            load_dotenv()
            groq_api_key = os.getenv("GROQ_API_KEY", "")
        
        if not groq_api_key:
            st.error("🚫 GROQ_API_KEY missing!")
            st.stop()
        
        return ChatGroq(
            groq_api_key=groq_api_key,
            model_name="openai/gpt-oss-120b",
            temperature=0.1,
            max_tokens=2000
        )
    except Exception as e:
        st.error(f"❌ LLM setup failed: {str(e)}")
        st.stop()

st.set_page_config(
    page_title="🛠️ Agentic Jira Maintenance Intelligence",
    layout="wide"
)

llm = get_llm()

MEMORY_FILE = "jira_agent_memory.txt"

@st.cache_data
def load_data():
    """Load CSV files from repo root (Streamlit Cloud ready)"""
    try:
        issues = pd.read_csv("issues.csv")
        projects = pd.read_csv("projects.csv")
        users = pd.read_csv("users.csv")

        issues.columns = issues.columns.str.strip()
        projects.columns = projects.columns.str.strip()
        users.columns = users.columns.str.strip()

        df = issues.merge(projects, on=["PROJECT_CODE", "PROJECT_NAME"], how="left")
        df = df.merge(users, left_on="ASSIGNEE", right_on="ID", how="left")

        df["CREATED"] = pd.to_datetime(df["CREATED"])
        df["RESOLVED_TS"] = pd.to_datetime(df["RESOLVED_TS"], errors="coerce")

        df["RESOLUTION_HOURS"] = (
            df["RESOLVED_TS"] - df["CREATED"]
        ).dt.total_seconds() / 3600

        st.success(f"Loaded {len(df)} issues from root directory")
        return df.fillna("N/A")
    except FileNotFoundError as e:
        st.error(f"❌ Missing CSV file: {str(e)}")
        st.info(" Ensure `issues.csv`, `projects.csv`, `users.csv` are in repo ROOT")
        st.stop()

issues_df = load_data()

@st.cache_resource
def build_vectorstore(df):
    """Build FAISS semantic search index"""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = []

    for _, row in df.iterrows():
        texts.append(
            f"Issue {row['KEY']} | {row['ISSUE_TYPE']} | "
            f"Project {row['PROJECT_NAME']} | "
            f"Severity {row['SEVERITY']} | Priority {row['PRIORITY']} | "
            f"Status {row['STATUS']} | Team {row['TEAM']} | "
            f"Env {row['ENVIRONMENT']} | Reopens {row['REOPEN_COUNT']} | "
            f"SLA {row['SLA_HOURS']}h | {row['SUMMARY']}"
        )

    embeddings = model.encode(texts).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    
    st.info("Vectorstore built successfully")
    return index, model

index, embedder = build_vectorstore(issues_df)

def semantic_retrieve(query, k=12):
    """Semantic search using FAISS"""
    q_emb = embedder.encode([query]).astype("float32")
    _, idx = index.search(q_emb, k)
    return issues_df.iloc[idx[0]].copy()

class JiraState(TypedDict):
    messages: Annotated[List[AIMessage | HumanMessage], add]
    raw_context: str
    metrics_summary: str
    action_plan: str
    viz_suggestions: List[str]
    risk_score: float
    plan: str


def planner_agent(state: JiraState):
    prompt = f"""
Decide execution steps for this Jira query:
{state['messages'][-1].content}

Return JSON with keys: retrieve, analyze, strategize, visualize
"""
    resp = llm.invoke([HumanMessage(content=prompt)])
    return {
        "messages": [AIMessage(content="🧠 PLANNER: Path decided")],
        "plan": resp.content
    }

def planner_router(state: JiraState):
    return "retriever" if "retrieve" in state["plan"].lower() else "visualizer"

def retriever_agent(state: JiraState):
    df = semantic_retrieve(state["messages"][-1].content, k=8)
    compact_df = df[["KEY", "PROJECT_NAME", "ISSUE_TYPE", "SEVERITY",
                    "PRIORITY", "STATUS", "TEAM", "ENVIRONMENT", "SLA_HOURS"]]
    context = compact_df.to_string(index=False)
    return {
        "messages": [AIMessage(content="📥 RETRIEVER: Compact context fetched")],
        "raw_context": context
    }

def analyzer_agent(state: JiraState):
    df = semantic_retrieve(state["messages"][-1].content, k=8)
    risk = (
        len(df[df["SEVERITY"].isin(["P0","P1"])]) * 0.4 +
        df["REOPEN_COUNT"].astype(int).mean() * 0.3 +
        (df["STATUS"] != "Done").mean() * 3
    )
    risk = round(min(10, risk), 2)

    prompt = f"""
Analyze Jira issues briefly.
Rules:
- Max 5 bullet points
- Each bullet ≤ 12 words  
- Focus on risk drivers only

Data:
{state['raw_context']}
"""
    resp = llm.invoke([HumanMessage(content=prompt)])
    return {
        "messages": [AIMessage(content="🔍 ANALYZER: Risk analysis complete")],
        "metrics_summary": resp.content,
        "risk_score": risk
    }

def strategist_agent(state: JiraState):
    prompt = f"""
Create a maintenance plan.
Constraints:
- Exactly 5 actions
- Each action ≤ 1 line
- No explanations

Risk Score: {state['risk_score']}
Analysis:
{state['metrics_summary']}
"""
    resp = llm.invoke([HumanMessage(content=prompt)])
    return {
        "messages": [AIMessage(content="🎯 STRATEGIST: Action plan generated")],
        "action_plan": resp.content
    }

def reflection_agent(state: JiraState):
    prompt = f"""
Refine the plan.
Rules:
- Improve clarity
- Remove redundancy
- Do NOT add new actions
- Keep total length same or shorter

Plan:
{state['action_plan']}
"""
    resp = llm.invoke([HumanMessage(content=prompt)])
    return {
        "messages": [AIMessage(content="🔁 REFLECTION: Plan refined")],
        "action_plan": resp.content
    }

def memory_agent(state: JiraState):
    """Store insights in root-level memory file"""
    try:
        with open(MEMORY_FILE, "a", encoding="utf-8") as f:
            f.write("\n---\n")
            f.write(state["metrics_summary"][:500])
    except Exception:
        pass  # Graceful fallback
    return {
        "messages": [AIMessage(content="🧠 MEMORY: Key insights stored")]
    }

def visualizer_agent(state: JiraState):
    return {
        "messages": [AIMessage(content="📊 VISUALIZER: Dashboard suggestions")],
        "viz_suggestions": [
            "P0/P1 incidents by project",
            "SLA risk by team", 
            "Reopen count heatmap",
            "Open issues by environment"
        ]
    }

def risk_router(state: JiraState):
    return "strategist" if state["risk_score"] > 7 else END


@st.cache_resource 
def build_pipeline():
    g = StateGraph(JiraState)
    
    g.add_node("planner", planner_agent)
    g.add_node("retriever", retriever_agent)
    g.add_node("analyzer", analyzer_agent)
    g.add_node("strategist", strategist_agent)
    g.add_node("reflection", reflection_agent)
    g.add_node("memory", memory_agent)
    g.add_node("visualizer", visualizer_agent)
    
    g.set_entry_point("planner")
    
    g.add_conditional_edges(
        "planner",
        planner_router,
        {"retriever": "retriever", "visualizer": "visualizer"}
    )
    
    g.add_edge("retriever", "analyzer")
    g.add_edge("analyzer", "strategist")
    g.add_edge("strategist", "reflection")
    g.add_edge("reflection", "memory")
    g.add_edge("memory", "visualizer")
    
    g.add_conditional_edges(
        "visualizer",
        risk_router,
        {"strategist": "strategist", END: END}
    )
    
    return g.compile()

pipeline = build_pipeline()

# === UI ===
st.title("🛠️ Agentic Jira Maintenance Intelligence")

# SIDEBAR
st.sidebar.header("🔎 Controls")

query = st.sidebar.text_area(
    "Jira Maintenance Query",
    "Analyze P0/P1 incidents and SLA risks by team and environment",
    height=120
)

run = st.sidebar.button("🚀 Run Agent Pipeline", use_container_width=True)

st.sidebar.divider()

project_filter = st.sidebar.multiselect(
    "Filter by Project", sorted(issues_df["PROJECT_NAME"].unique())
)
team_filter = st.sidebar.multiselect(
    "Filter by Team", sorted(issues_df["TEAM"].unique())
)
env_filter = st.sidebar.multiselect(
    "Filter by Environment", sorted(issues_df["ENVIRONMENT"].unique())
)

#filter data
filtered_df = issues_df.copy()
if project_filter: 
    filtered_df = filtered_df[filtered_df["PROJECT_NAME"].isin(project_filter)]
if team_filter:
    filtered_df = filtered_df[filtered_df["TEAM"].isin(team_filter)]
if env_filter:
    filtered_df = filtered_df[filtered_df["ENVIRONMENT"].isin(env_filter)]


if run:
    with st.spinner("🤖 Agents collaborating..."):
        result = pipeline.invoke({
            "messages": [HumanMessage(content=query)],
            "raw_context": "", "metrics_summary": "", "action_plan": "",
            "viz_suggestions": [], "risk_score": 0.0, "plan": ""
        })

    st.divider()

    # AGENT TRACE
    st.subheader("🧠 Agent Execution Trace")
    for msg in result["messages"]:
        st.markdown(f"• {msg.content}")

    # RISK OVERVIEW
    st.subheader("⚠️ Risk Overview")
    risk = result["risk_score"]

    if risk <= 3:
        level = "🟢 Low Risk"
    elif risk <= 7:
        level = "🟠 Medium Risk"
    else:
        level = "🔴 High Risk"

    col1, col2 = st.columns([3, 1])
    col1.metric("Overall Risk Score", risk)
    col2.metric("Level", level)
    
    st.progress(risk / 10)

    #action plan
    st.subheader("🎯 Maintenance Action Plan")
    st.success(result["action_plan"])

    # DASHBOARD SUGGESTIONS
    st.subheader("📊 Recommended Dashboards")
    for v in result["viz_suggestions"]:
        st.info(v)

#ANALYTICS DASHBOARD
st.divider()
st.subheader("📈 Jira Reliability Analytics")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Issues", len(filtered_df))
k2.metric("P0/P1 Issues", len(filtered_df[filtered_df["SEVERITY"].isin(["P0", "P1"])]) )
k3.metric("Open Issues", len(filtered_df[filtered_df["STATUS"] != "Done"]))
k4.metric("Avg Resolution (hrs)", round(
    filtered_df["RESOLUTION_HOURS"].replace("N/A", np.nan).astype(float).mean(), 2))

#CHARTS
c1, c2 = st.columns(2)
c1.plotly_chart(px.bar(filtered_df, x="PROJECT_NAME", color="SEVERITY", 
                      title="Severity by Project", barmode="group"), 
                use_container_width=True)
c2.plotly_chart(px.bar(filtered_df, x="TEAM", color="STATUS", 
                      title="Status by Team", barmode="stack"), 
                use_container_width=True)

st.plotly_chart(px.bar(filtered_df, x="ENVIRONMENT", color="ISSUE_TYPE",
                      title="Issues by Environment", barmode="group"), 
                use_container_width=True)


