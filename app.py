"""
=============================================================================
   AI Career Copilot MVP  —  Single-File Application
   FastAPI (REST APIs) + Streamlit (UI) + SQLite + FAISS + RAG
   Python 3.10 | Free & Open-Source Only | Groq Free Tier
=============================================================================

Architecture Overview:
  - All business logic lives as plain Python functions (Section 3-8).
  - FastAPI endpoints (Section 9) expose REST APIs that call those functions.
  - Streamlit UI (Section 10) calls the same functions directly (no HTTP
    overhead when running in a single process).
  - At startup: FastAPI is launched in a daemon thread, Streamlit runs as
    the main process.

Quick Start:
  $ pip install -r requirements.txt
  $ cp .env.example .env          # add your GROQ_API_KEY
  $ streamlit run app.py          # starts both UI + API
  $ uvicorn app:api --port 8000   # API-only mode (optional)
=============================================================================
"""

# ============================================================================
# SECTION 1: IMPORTS
# ============================================================================

import os
import sys
import json
import logging
import sqlite3
import hashlib
import secrets
import tempfile
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any

# --- Env / Config ---
from dotenv import load_dotenv

# --- Data ---
import numpy as np

# --- PDF Parsing ---
import fitz  # PyMuPDF

# --- Embeddings & Vector Store ---
from sentence_transformers import SentenceTransformer
import faiss

# --- LangChain (Community Edition) ---
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain_community.llms import Groq as GroqLLM
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# --- Auth ---
import jwt

# --- FastAPI ---
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# --- Streamlit ---
import streamlit as st

# ============================================================================
# SECTION 2: CONFIGURATION & LOGGING
# ============================================================================

load_dotenv()

# --- Project Paths ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "career_copilot.db"
FAISS_DIR = DATA_DIR / "faiss_index"
FAISS_DIR.mkdir(exist_ok=True)

# --- App Settings ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
JWT_SECRET = os.getenv("JWT_SECRET", secrets.token_hex(32))
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRY_HOURS = int(os.getenv("JWT_EXPIRY_HOURS", "24"))
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")
FASTAPI_PORT = int(os.getenv("FASTAPI_PORT", "8000"))

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-18s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(DATA_DIR / "app.log", mode="a"),
    ],
)
logger = logging.getLogger("career_copilot")


# ============================================================================
# SECTION 3: CENTRALIZED ERROR HANDLING
# ============================================================================

class AppError(Exception):
    """Base application error with user-friendly message and HTTP status."""

    def __init__(self, message: str, status_code: int = 500, detail: str = ""):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.detail = detail or message
        logger.error(f"AppError({status_code}): {message} | {detail}")


def handle_error(func):
    """Decorator that wraps functions with centralized error handling."""

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except AppError:
            raise
        except Exception as exc:
            logger.exception(f"Unhandled error in {func.__name__}")
            raise AppError(
                message="An unexpected error occurred. Please try again.",
                status_code=500,
                detail=str(exc),
            )

    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper


# ============================================================================
# SECTION 4: DATABASE LAYER (SQLite)
# ============================================================================

def get_db() -> sqlite3.Connection:
    """Return a new SQLite connection with row factory."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create all tables if they don't exist."""
    conn = get_db()
    cursor = conn.cursor()

    # Users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Resumes table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS resumes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            raw_text TEXT NOT NULL,
            chunks_json TEXT,
            uploaded_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    # Job descriptions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS job_descriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    # Chat history table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            resume_id INTEGER,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    # Analyses table (skill-gap, roadmaps, interviews, scores)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            resume_id INTEGER NOT NULL,
            job_id INTEGER,
            analysis_type TEXT NOT NULL,
            result_json TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    conn.commit()
    conn.close()
    logger.info("Database initialized successfully")


# ============================================================================
# SECTION 5: AUTHENTICATION (JWT)
# ============================================================================

def hash_password(password: str) -> str:
    """Hash a password using SHA-256 with salt."""
    salt = secrets.token_hex(16)
    hashed = hashlib.sha256(f"{salt}{password}".encode()).hexdigest()
    return f"{salt}:{hashed}"


def verify_password(password: str, stored_hash: str) -> bool:
    """Verify a password against the stored salt:hash."""
    salt, hashed = stored_hash.split(":")
    computed = hashlib.sha256(f"{salt}{password}".encode()).hexdigest()
    return secrets.compare_digest(computed, hashed)


def create_token(user_id: int, username: str) -> str:
    """Generate a JWT token for the given user."""
    payload = {
        "user_id": user_id,
        "username": username,
        "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRY_HOURS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> Dict[str, Any]:
    """Decode and validate a JWT token. Raises AppError on failure."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise AppError("Token has expired. Please log in again.", 401)
    except jwt.InvalidTokenError:
        raise AppError("Invalid token.", 401)


@handle_error
def register_user(username: str, email: str, password: str) -> Dict[str, Any]:
    """Register a new user. Returns token on success."""
    conn = get_db()
    try:
        pw_hash = hash_password(password)
        cursor = conn.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            (username, email, pw_hash),
        )
        conn.commit()
        user_id = cursor.lastrowid
        token = create_token(user_id, username)
        logger.info(f"User registered: {username}")
        return {"user_id": user_id, "username": username, "token": token}
    except sqlite3.IntegrityError:
        raise AppError("Username or email already exists.", 409)
    finally:
        conn.close()


@handle_error
def login_user(username: str, password: str) -> Dict[str, Any]:
    """Authenticate a user. Returns token on success."""
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT id, username, password_hash FROM users WHERE username = ?",
            (username,),
        ).fetchone()
        if not row or not verify_password(password, row["password_hash"]):
            raise AppError("Invalid username or password.", 401)
        token = create_token(row["id"], row["username"])
        logger.info(f"User logged in: {username}")
        return {"user_id": row["id"], "username": row["username"], "token": token}
    finally:
        conn.close()


def get_current_user(authorization: str = "") -> Dict[str, Any]:
    """Extract and validate user from Authorization header (Bearer token)."""
    if not authorization or not authorization.startswith("Bearer "):
        raise AppError("Missing or invalid Authorization header.", 401)
    token = authorization.split(" ", 1)[1]
    return decode_token(token)


# ============================================================================
# SECTION 6: EMBEDDING & VECTOR STORE (FAISS)
# ============================================================================

# Lazy-loaded singleton for the embedding model
_embedding_model: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    """Load and cache the sentence-transformer model."""
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logger.info("Embedding model loaded successfully")
    return _embedding_model


def generate_embeddings(texts: List[str]) -> np.ndarray:
    """Generate embeddings for a list of text chunks."""
    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=False)
    return np.array(embeddings, dtype=np.float32)


def create_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Create a FAISS inner-product index from embeddings."""
    # Normalize for cosine similarity via inner product
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def save_faiss_index(index: faiss.Index, resume_id: int):
    """Persist FAISS index and its chunk metadata to disk."""
    index_path = FAISS_DIR / f"resume_{resume_id}.index"
    faiss.write_index(index, str(index_path))
    logger.info(f"FAISS index saved for resume_id={resume_id}")


def load_faiss_index(resume_id: int) -> Optional[faiss.Index]:
    """Load a FAISS index from disk. Returns None if not found."""
    index_path = FAISS_DIR / f"resume_{resume_id}.index"
    if index_path.exists():
        return faiss.read_index(str(index_path))
    return None


def search_faiss(index: faiss.Index, query_embedding: np.ndarray, top_k: int = 5):
    """Search the FAISS index for the most similar chunks."""
    query_embedding = np.array([query_embedding], dtype=np.float32)
    faiss.normalize_L2(query_embedding)
    scores, indices = index.search(query_embedding, top_k)
    return scores[0], indices[0]


# ============================================================================
# SECTION 7: PDF PARSING & SEMANTIC CHUNKING
# ============================================================================

@handle_error
def parse_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF file using PyMuPDF."""
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        if not text.strip():
            raise AppError("Could not extract text from PDF. It may be image-based.", 400)
        logger.info(f"PDF parsed: {len(text)} characters extracted")
        return text
    except AppError:
        raise
    except Exception as e:
        raise AppError(f"Failed to parse PDF: {str(e)}", 400)


def semantic_chunk(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> List[str]:
    """
    Split text into semantic chunks using RecursiveCharacterTextSplitter.
    Tries to break at paragraph/sentence boundaries for coherent chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_text(text)
    logger.info(f"Text chunked into {len(chunks)} segments")
    return chunks


# ============================================================================
# SECTION 8: BUSINESS LOGIC (RAG Pipeline + Analysis Features)
# ============================================================================

# --- Groq LLM Helper ---
#_groq_llm: Optional[GroqLLM] = None
_groq_llm: Optional[ChatGroq] = None


#def get_groq_llm() -> GroqLLM:
def get_groq_llm() -> ChatGroq:
    """Initialize and cache the Groq LLM client."""
    global _groq_llm
    if _groq_llm is None:
        if not GROQ_API_KEY:
            raise AppError(
                "GROQ_API_KEY not set. Add it to your .env file.",
                500,
                "Missing Groq API key",
            )
        #_groq_llm = GroqLLM(
            groq_api_key=GROQ_API_KEY,
            model_name=GROQ_MODEL,
            temperature=0.3,
            max_tokens=2048,
        #)
        _groq_llm = ChatGroq(groq_api_key=GROQ_API_KEY,
            model_name=GROQ_MODEL,
            temperature=0.3,
            max_tokens=2048,
        )
        logger.info(f"Groq LLM initialized: {GROQ_MODEL}")
    return _groq_llm


def ask_llm(prompt: str) -> str:
    """Send a prompt to Groq LLM and return the response text."""
    llm = get_groq_llm()
    try:
        response = llm.invoke(prompt)
        #return response.strip()
        return response.content.strip()
    except Exception as e:
        logger.error(f"LLM invocation failed: {e}")
        raise AppError(f"LLM call failed: {str(e)}", 502)


# --- 8a. Resume Upload & Processing ---

@handle_error
def process_resume(user_id: int, filename: str, file_bytes: bytes) -> Dict[str, Any]:
    """
    Full resume processing pipeline:
    1. Parse PDF
    2. Semantic chunking
    3. Generate embeddings
    4. Build & save FAISS index
    5. Store metadata in SQLite
    """
    raw_text = parse_pdf(file_bytes)
    chunks = semantic_chunk(raw_text)
    embeddings = generate_embeddings(chunks)
    index = create_faiss_index(embeddings)

    # Save to DB
    conn = get_db()
    try:
        cursor = conn.execute(
            "INSERT INTO resumes (user_id, filename, raw_text, chunks_json) VALUES (?, ?, ?, ?)",
            (user_id, filename, raw_text, json.dumps(chunks)),
        )
        conn.commit()
        resume_id = cursor.lastrowid
    finally:
        conn.close()

    # Save FAISS index
    save_faiss_index(index, resume_id)

    logger.info(f"Resume processed: id={resume_id}, chunks={len(chunks)}")
    return {
        "resume_id": resume_id,
        "filename": filename,
        "char_count": len(raw_text),
        "chunk_count": len(chunks),
    }


# --- 8b. Resume Q&A Chat (Simple RAG) ---

@handle_error
def resume_qa(user_id: int, resume_id: int, question: str) -> Dict[str, Any]:
    """
    Answer a question about a resume using RAG:
    1. Embed the question
    2. Search FAISS for relevant chunks
    3. Build a prompt with retrieved context
    4. Send to Groq LLM for answer
    """
    # Load chunks and index
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT chunks_json FROM resumes WHERE id = ? AND user_id = ?",
            (resume_id, user_id),
        ).fetchone()
    finally:
        conn.close()

    if not row:
        raise AppError("Resume not found.", 404)

    chunks = json.loads(row["chunks_json"])
    index = load_faiss_index(resume_id)
    if index is None:
        raise AppError("Resume index not found. Please re-upload.", 404)

    # Embed question and search
    query_embedding = generate_embeddings([question])[0]
    scores, indices = search_faiss(index, query_embedding, top_k=5)

    # Gather relevant context
    context_parts = []
    for idx in indices:
        if 0 <= idx < len(chunks):
            context_parts.append(chunks[idx])
    context = "\n\n---\n\n".join(context_parts)

    # Build RAG prompt
    prompt = f"""You are an AI Career Copilot. Answer the user's question based on the resume context below.
If the answer is not found in the context, say so clearly.

RESUME CONTEXT:
{context}

USER QUESTION: {question}

ANSWER:"""

    answer = ask_llm(prompt)

    # Save to chat history
    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO chat_history (user_id, resume_id, question, answer) VALUES (?, ?, ?, ?)",
            (user_id, resume_id, question, answer),
        )
        conn.commit()
    finally:
        conn.close()

    return {"question": question, "answer": answer, "context_chunks_used": len(context_parts)}


# --- 8c. Job Description Matching ---

@handle_error
def match_job(user_id: int, resume_id: int, job_description: str, job_title: str = "Untitled") -> Dict[str, Any]:
    """
    Match a resume against a job description:
    1. Retrieve relevant resume chunks
    2. Ask LLM to compare and produce a fit analysis
    """
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT raw_text, chunks_json FROM resumes WHERE id = ? AND user_id = ?",
            (resume_id, user_id),
        ).fetchone()
    finally:
        conn.close()

    if not row:
        raise AppError("Resume not found.", 404)

    chunks = json.loads(row["chunks_json"])
    index = load_faiss_index(resume_id)
    if index is None:
        raise AppError("Resume index not found.", 404)

    # Embed job description and find matching resume sections
    job_embedding = generate_embeddings([job_description])[0]
    scores, indices = search_faiss(index, job_embedding, top_k=8)

    context_parts = [chunks[i] for i in indices if 0 <= i < len(chunks)]
    context = "\n\n---\n\n".join(context_parts)
    avg_score = float(np.mean(scores[scores > 0])) if np.any(scores > 0) else 0.0

    # Save job description
    conn = get_db()
    try:
        cursor = conn.execute(
            "INSERT INTO job_descriptions (user_id, title, description) VALUES (?, ?, ?)",
            (user_id, job_title, job_description),
        )
        conn.commit()
        job_id = cursor.lastrowid
    finally:
        conn.close()

    prompt = f"""You are an expert career advisor. Compare the resume sections below with the job description.
Provide:
1. Key matching skills and experiences
2. Missing skills or experiences
3. Overall compatibility assessment
4. Specific suggestions for improvement

JOB DESCRIPTION:
{job_description}

RESUME SECTIONS:
{context}

DETAILED MATCH ANALYSIS:"""

    analysis = ask_llm(prompt)

    # Save analysis
    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO analyses (user_id, resume_id, job_id, analysis_type, result_json) VALUES (?, ?, ?, ?, ?)",
            (user_id, resume_id, job_id, "job_match", json.dumps({"analysis": analysis, "similarity_score": avg_score})),
        )
        conn.commit()
    finally:
        conn.close()

    return {
        "job_id": job_id,
        "similarity_score": round(avg_score, 4),
        "analysis": analysis,
    }


# --- 8d. Skill-Gap Analysis ---

@handle_error
def skill_gap_analysis(user_id: int, resume_id: int, job_description: str) -> Dict[str, Any]:
    """Analyze the gap between resume skills and job requirements."""
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT raw_text FROM resumes WHERE id = ? AND user_id = ?",
            (resume_id, user_id),
        ).fetchone()
    finally:
        conn.close()

    if not row:
        raise AppError("Resume not found.", 404)

    resume_text = row["raw_text"][:3000]  # Truncate for prompt limits

    prompt = f"""You are an expert technical recruiter. Perform a detailed skill-gap analysis.

RESUME (excerpt):
{resume_text}

TARGET JOB DESCRIPTION:
{job_description}

Provide your analysis in this format:
1. SKILLS YOU HAVE (that match the job)
2. SKILLS YOU'RE MISSING (required by the job but not in your resume)
3. PARTIAL SKILLS (skills you have some experience with but need deeper expertise)
4. PRIORITY RECOMMENDATIONS (which gaps to close first and why)

DETAILED SKILL-GAP ANALYSIS:"""

    analysis = ask_llm(prompt)

    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO analyses (user_id, resume_id, analysis_type, result_json) VALUES (?, ?, ?, ?)",
            (user_id, resume_id, "skill_gap", json.dumps({"analysis": analysis})),
        )
        conn.commit()
    finally:
        conn.close()

    return {"analysis": analysis}


# --- 8e. Learning Roadmap Generation ---

@handle_error
def generate_roadmap(user_id: int, resume_id: int, target_role: str) -> Dict[str, Any]:
    """Generate a personalized learning roadmap for a target role."""
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT raw_text FROM resumes WHERE id = ? AND user_id = ?",
            (resume_id, user_id),
        ).fetchone()
    finally:
        conn.close()

    if not row:
        raise AppError("Resume not found.", 404)

    resume_text = row["raw_text"][:3000]

    prompt = f"""You are a career mentor and learning strategist. Based on the person's current resume and their target role, create a practical learning roadmap.

CURRENT RESUME (excerpt):
{resume_text}

TARGET ROLE: {target_role}

Create a structured roadmap with:
1. PHASE 1 - Foundation (0-2 months): Core skills to build
2. PHASE 2 - Intermediate (2-4 months): Building depth
3. PHASE 3 - Advanced (4-6 months): Specialization
4. PHASE 4 - Portfolio & Networking (6-8 months): Real-world application

For each phase, include:
- Specific topics/concepts to learn
- Recommended free resources (courses, books, YouTube channels)
- Mini-projects to build
- Metrics to track progress

LEARNING ROADMAP:"""

    roadmap = ask_llm(prompt)

    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO analyses (user_id, resume_id, analysis_type, result_json) VALUES (?, ?, ?, ?)",
            (user_id, resume_id, "roadmap", json.dumps({"target_role": target_role, "roadmap": roadmap})),
        )
        conn.commit()
    finally:
        conn.close()

    return {"target_role": target_role, "roadmap": roadmap}


# --- 8f. Mock Interview Question Generation ---

@handle_error
def generate_interview_questions(
    user_id: int, resume_id: int, target_role: str, num_questions: int = 10
) -> Dict[str, Any]:
    """Generate mock interview questions tailored to the resume and target role."""
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT raw_text FROM resumes WHERE id = ? AND user_id = ?",
            (resume_id, user_id),
        ).fetchone()
    finally:
        conn.close()

    if not row:
        raise AppError("Resume not found.", 404)

    resume_text = row["raw_text"][:3000]

    prompt = f"""You are a technical interview coach. Generate {num_questions} mock interview questions based on this resume and target role.

CURRENT RESUME (excerpt):
{resume_text}

TARGET ROLE: {target_role}

Generate a mix of:
- Behavioral questions (STAR method)
- Technical questions (based on resume skills)
- Situational questions (real-world scenarios)
- Culture-fit questions

For each question, provide:
1. The question
2. What the interviewer is looking for
3. A brief tip on how to answer well
4. Difficulty level (Easy/Medium/Hard)

MOCK INTERVIEW QUESTIONS:"""

    questions = ask_llm(prompt)

    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO analyses (user_id, resume_id, analysis_type, result_json) VALUES (?, ?, ?, ?)",
            (user_id, resume_id, "interview", json.dumps({"target_role": target_role, "questions": questions})),
        )
        conn.commit()
    finally:
        conn.close()

    return {"target_role": target_role, "questions": questions}


# --- 8g. Recruiter Fit Score ---

@handle_error
def recruiter_fit_score(
    user_id: int, resume_id: int, job_description: str
) -> Dict[str, Any]:
    """
    Generate a comprehensive recruiter fit score with detailed breakdown.
    Uses both semantic similarity and LLM-based assessment.
    """
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT raw_text, chunks_json FROM resumes WHERE id = ? AND user_id = ?",
            (resume_id, user_id),
        ).fetchone()
    finally:
        conn.close()

    if not row:
        raise AppError("Resume not found.", 404)

    chunks = json.loads(row["chunks_json"])
    index = load_faiss_index(resume_id)
    if index is None:
        raise AppError("Resume index not found.", 404)

    # Semantic similarity component
    job_embedding = generate_embeddings([job_description])[0]
    scores, _ = search_faiss(index, job_embedding, top_k=10)
    semantic_score = float(np.mean(scores[scores > 0])) * 100 if np.any(scores > 0) else 0.0
    semantic_score = min(semantic_score, 100)

    resume_text = row["raw_text"][:3000]

    prompt = f"""You are a senior technical recruiter at a top company. Evaluate this resume against the job description.

RESUME (excerpt):
{resume_text}

JOB DESCRIPTION:
{job_description}

Provide scores (0-100) for each category:
1. SKILLS MATCH: How well do the candidate's skills match the job requirements?
2. EXPERIENCE RELEVANCE: How relevant is their experience?
3. EDUCATION FIT: How well does their education align?
4. OVERALL IMPRESSION: General recruiter impression

Also provide:
- A one-paragraph summary of the candidate's strengths
- A one-paragraph summary of concerns
- A HIRE/NO-HIRE recommendation with confidence level

Format your response clearly with labeled scores and sections.

RECRUITER EVALUATION:"""

    evaluation = ask_llm(prompt)

    # Save analysis
    result = {
        "semantic_similarity_score": round(semantic_score, 1),
        "evaluation": evaluation,
    }
    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO analyses (user_id, resume_id, analysis_type, result_json) VALUES (?, ?, ?, ?)",
            (user_id, resume_id, "fit_score", json.dumps(result)),
        )
        conn.commit()
    finally:
        conn.close()

    return result


# --- 8h. Utility: List Resumes / Chat History ---

@handle_error
def list_resumes(user_id: int) -> List[Dict[str, Any]]:
    """List all resumes for a user."""
    conn = get_db()
    try:
        rows = conn.execute(
            #"SELECT id, filename, char_count(raw_text) as chars, uploaded_at FROM resumes WHERE user_id = ? ORDER BY uploaded_at DESC",
            "SELECT id, filename, length(raw_text) as chars, uploaded_at FROM resumes WHERE user_id = ? ORDER BY uploaded_at DESC",
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


@handle_error
def get_chat_history(user_id: int, resume_id: Optional[int] = None, limit: int = 50) -> List[Dict[str, Any]]:
    """Get chat history for a user, optionally filtered by resume."""
    conn = get_db()
    try:
        if resume_id:
            rows = conn.execute(
                "SELECT question, answer, created_at FROM chat_history WHERE user_id = ? AND resume_id = ? ORDER BY created_at DESC LIMIT ?",
                (user_id, resume_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT question, answer, resume_id, created_at FROM chat_history WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ============================================================================
# SECTION 9: FASTAPI REST API ENDPOINTS
# ============================================================================

api = FastAPI(
    title="AI Career Copilot API",
    version="1.0.0",
    description="REST API for resume analysis, job matching, and career guidance",
)

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@api.on_event("startup")
def on_startup():
    """Initialize DB and embedding model on FastAPI startup."""
    init_db()
    # Pre-load embedding model
    get_embedding_model()
    logger.info("FastAPI startup complete")


# --- Health Check ---

@api.get("/health")
def health_check():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


# --- Auth Endpoints ---

@api.post("/auth/register")
def api_register(payload: dict):
    """Register a new user."""
    username = payload.get("username", "").strip()
    email = payload.get("email", "").strip()
    password = payload.get("password", "")
    if not all([username, email, password]):
        raise HTTPException(400, "username, email, and password are required")
    if len(password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters")
    try:
        return register_user(username, email, password)
    except AppError as e:
        raise HTTPException(e.status_code, e.message)


@api.post("/auth/login")
def api_login(payload: dict):
    """Login and receive a JWT token."""
    username = payload.get("username", "").strip()
    password = payload.get("password", "")
    if not all([username, password]):
        raise HTTPException(400, "username and password are required")
    try:
        return login_user(username, password)
    except AppError as e:
        raise HTTPException(e.status_code, e.message)


# --- Resume Endpoints ---

@api.post("/resumes/upload")
def api_upload_resume(file: UploadFile = File(...), authorization: str = Header("")):
    """Upload and process a resume PDF."""
    user = get_current_user(authorization)
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")
    try:
        file_bytes = file.file.read()
        return process_resume(user["user_id"], file.filename, file_bytes)
    except AppError as e:
        raise HTTPException(e.status_code, e.message)


@api.get("/resumes")
def api_list_resumes(authorization: str = Header("")):
    """List all resumes for the current user."""
    user = get_current_user(authorization)
    try:
        return list_resumes(user["user_id"])
    except AppError as e:
        raise HTTPException(e.status_code, e.message)


# --- Chat Q&A Endpoint ---

@api.post("/chat/qa")
def api_resume_qa(payload: dict, authorization: str = Header("")):
    """Ask a question about a resume."""
    user = get_current_user(authorization)
    resume_id = payload.get("resume_id")
    question = payload.get("question", "").strip()
    if not resume_id or not question:
        raise HTTPException(400, "resume_id and question are required")
    try:
        return resume_qa(user["user_id"], resume_id, question)
    except AppError as e:
        raise HTTPException(e.status_code, e.message)


@api.get("/chat/history")
def api_chat_history(authorization: str = Header(""), resume_id: Optional[int] = None):
    """Get chat history."""
    user = get_current_user(authorization)
    try:
        return get_chat_history(user["user_id"], resume_id)
    except AppError as e:
        raise HTTPException(e.status_code, e.message)


# --- Analysis Endpoints ---

@api.post("/analysis/job-match")
def api_job_match(payload: dict, authorization: str = Header("")):
    """Match a resume against a job description."""
    user = get_current_user(authorization)
    resume_id = payload.get("resume_id")
    job_description = payload.get("job_description", "").strip()
    job_title = payload.get("job_title", "Untitled")
    if not resume_id or not job_description:
        raise HTTPException(400, "resume_id and job_description are required")
    try:
        return match_job(user["user_id"], resume_id, job_description, job_title)
    except AppError as e:
        raise HTTPException(e.status_code, e.message)


@api.post("/analysis/skill-gap")
def api_skill_gap(payload: dict, authorization: str = Header("")):
    """Analyze skill gaps between resume and job."""
    user = get_current_user(authorization)
    resume_id = payload.get("resume_id")
    job_description = payload.get("job_description", "").strip()
    if not resume_id or not job_description:
        raise HTTPException(400, "resume_id and job_description are required")
    try:
        return skill_gap_analysis(user["user_id"], resume_id, job_description)
    except AppError as e:
        raise HTTPException(e.status_code, e.message)


@api.post("/analysis/roadmap")
def api_roadmap(payload: dict, authorization: str = Header("")):
    """Generate a learning roadmap."""
    user = get_current_user(authorization)
    resume_id = payload.get("resume_id")
    target_role = payload.get("target_role", "").strip()
    if not resume_id or not target_role:
        raise HTTPException(400, "resume_id and target_role are required")
    try:
        return generate_roadmap(user["user_id"], resume_id, target_role)
    except AppError as e:
        raise HTTPException(e.status_code, e.message)


@api.post("/analysis/interview")
def api_interview(payload: dict, authorization: str = Header("")):
    """Generate mock interview questions."""
    user = get_current_user(authorization)
    resume_id = payload.get("resume_id")
    target_role = payload.get("target_role", "").strip()
    num_questions = payload.get("num_questions", 10)
    if not resume_id or not target_role:
        raise HTTPException(400, "resume_id and target_role are required")
    try:
        return generate_interview_questions(user["user_id"], resume_id, target_role, num_questions)
    except AppError as e:
        raise HTTPException(e.status_code, e.message)


@api.post("/analysis/fit-score")
def api_fit_score(payload: dict, authorization: str = Header("")):
    """Calculate recruiter fit score."""
    user = get_current_user(authorization)
    resume_id = payload.get("resume_id")
    job_description = payload.get("job_description", "").strip()
    if not resume_id or not job_description:
        raise HTTPException(400, "resume_id and job_description are required")
    try:
        return recruiter_fit_score(user["user_id"], resume_id, job_description)
    except AppError as e:
        raise HTTPException(e.status_code, e.message)


# --- Global Exception Handler for FastAPI ---

@api.exception_handler(AppError)
async def app_error_handler(request, exc: AppError):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.message, "detail": exc.detail})


# ============================================================================
# SECTION 10: STREAMLIT UI
# ============================================================================

def run_streamlit():
    """Main Streamlit application."""

    st.set_page_config(
        page_title="AI Career Copilot",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # --- Custom CSS ---
    st.markdown("""
    <style>
        .main-header { font-size: 2.5rem; font-weight: 700; color: #1E88E5; }
        .sub-header { font-size: 1.3rem; font-weight: 600; color: #424242; }
        .success-box { padding: 1rem; border-radius: 0.5rem; background: #E8F5E9; border-left: 4px solid #43A047; }
        .info-box { padding: 1rem; border-radius: 0.5rem; background: #E3F2FD; border-left: 4px solid #1E88E5; }
        .warning-box { padding: 1rem; border-radius: 0.5rem; background: #FFF8E1; border-left: 4px solid #FFA000; }
        .analysis-card { padding: 1.5rem; border-radius: 0.75rem; background: #FAFAFA; border: 1px solid #E0E0E0; margin: 1rem 0; }
        .score-circle { font-size: 3rem; font-weight: 700; text-align: center; }
    </style>
    """, unsafe_allow_html=True)

    # --- Session State Initialization ---
    if "token" not in st.session_state:
        st.session_state.token = None
    if "username" not in st.session_state:
        st.session_state.username = None
    if "user_id" not in st.session_state:
        st.session_state.user_id = None
    if "current_resume_id" not in st.session_state:
        st.session_state.current_resume_id = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Initialize DB on first Streamlit run
    init_db()

    # --- Auth Pages ---
    if not st.session_state.token:
        st.markdown('<p class="main-header">🚀 AI Career Copilot</p>', unsafe_allow_html=True)
        st.markdown("Your intelligent career companion — powered by AI")

        tab_reg, tab_login = st.tabs(["Register", "Login"])

        with tab_reg:
            with st.form("register_form"):
                reg_user = st.text_input("Username", key="reg_username")
                reg_email = st.text_input("Email", key="reg_email")
                reg_pass = st.text_input("Password (6+ chars)", type="password", key="reg_password")
                reg_submit = st.form_submit_button("Create Account")
                if reg_submit:
                    try:
                        result = register_user(reg_user, reg_email, reg_pass)
                        st.session_state.token = result["token"]
                        st.session_state.username = result["username"]
                        st.session_state.user_id = result["user_id"]
                        st.success("Account created! Welcome aboard 🎉")
                        st.rerun()
                    except AppError as e:
                        st.error(e.message)

        with tab_login:
            with st.form("login_form"):
                log_user = st.text_input("Username", key="log_username")
                log_pass = st.text_input("Password", type="password", key="log_password")
                log_submit = st.form_submit_button("Login")
                if log_submit:
                    try:
                        result = login_user(log_user, log_pass)
                        st.session_state.token = result["token"]
                        st.session_state.username = result["username"]
                        st.session_state.user_id = result["user_id"]
                        st.success(f"Welcome back, {result['username']}! 🎉")
                        st.rerun()
                    except AppError as e:
                        st.error(e.message)
        return

    # --- Main App (Logged In) ---
    st.sidebar.markdown(f"👤 **{st.session_state.username}**")
    if st.sidebar.button("Logout"):
        st.session_state.token = None
        st.session_state.username = None
        st.session_state.user_id = None
        st.session_state.current_resume_id = None
        st.session_state.chat_history = []
        st.rerun()

    st.sidebar.divider()

    # Sidebar navigation
    page = st.sidebar.radio(
        "Navigate",
        ["📄 Resume Upload", "💬 Resume Q&A", "🎯 Job Matching",
         "📊 Skill-Gap Analysis", "🗺️ Learning Roadmap",
         "🎤 Mock Interview", "📈 Fit Score"],
    )

    # Always show resume selector in sidebar
    st.sidebar.divider()
    st.sidebar.subheader("Active Resume")
    try:
        resumes = list_resumes(st.session_state.user_id)
        if resumes:
            resume_options = {f"ID:{r['id']} - {r['filename']}": r["id"] for r in resumes}
            selected = st.sidebar.selectbox("Select a resume", list(resume_options.keys()))
            st.session_state.current_resume_id = resume_options[selected]
        else:
            st.sidebar.info("Upload a resume first!")
    except Exception:
        st.sidebar.warning("Could not load resumes")

    # --- Page: Resume Upload ---
    if page == "📄 Resume Upload":
        st.markdown('<p class="main-header">📄 Resume Upload</p>', unsafe_allow_html=True)
        st.markdown("Upload your resume PDF and we'll parse, chunk, embed, and index it for AI analysis.")

        uploaded = st.file_uploader("Choose a PDF resume", type=["pdf"])
        if uploaded is not None:
            if st.button("🔍 Process Resume", type="primary"):
                with st.spinner("Processing resume... This may take a moment on first run (model download)."):
                    try:
                        file_bytes = uploaded.read()
                        result = process_resume(
                            st.session_state.user_id, uploaded.name, file_bytes
                        )
                        st.session_state.current_resume_id = result["resume_id"]
                        st.markdown('<div class="success-box">', unsafe_allow_html=True)
                        st.success(f"Resume processed successfully!")
                        st.json(result)
                        st.markdown('</div>', unsafe_allow_html=True)
                    except AppError as e:
                        st.error(f"Error: {e.message}")

        # Show uploaded resumes
        st.divider()
        st.subheader("Your Resumes")
        try:
            resumes = list_resumes(st.session_state.user_id)
            if resumes:
                for r in resumes:
                    with st.expander(f"📄 {r['filename']} (uploaded {r['uploaded_at']})"):
                        st.write(f"**Resume ID:** {r['id']}")
                        st.write(f"**Characters:** {r['chars']}")
            else:
                st.info("No resumes uploaded yet. Upload your first one above!")
        except Exception as e:
            st.error(f"Could not load resumes: {e}")

    # --- Page: Resume Q&A Chat ---
    elif page == "💬 Resume Q&A":
        st.markdown('<p class="main-header">💬 Resume Q&A Chat</p>', unsafe_allow_html=True)
        st.markdown("Ask anything about your resume — the AI uses RAG to give context-aware answers.")

        if not st.session_state.current_resume_id:
            st.warning("Please upload and select a resume first!")
            return

        st.info(f"Chatting about Resume ID: {st.session_state.current_resume_id}")

        # Display chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat input
        if prompt := st.chat_input("Ask about your resume..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        result = resume_qa(
                            st.session_state.user_id,
                            st.session_state.current_resume_id,
                            prompt,
                        )
                        answer = result["answer"]
                        st.markdown(answer)
                        st.caption(f"Retrieved {result['context_chunks_used']} relevant chunks")
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    except AppError as e:
                        error_msg = f"Error: {e.message}"
                        st.error(error_msg)
                        st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

    # --- Page: Job Matching ---
    elif page == "🎯 Job Matching":
        st.markdown('<p class="main-header">🎯 Job Description Matching</p>', unsafe_allow_html=True)
        st.markdown("Paste a job description and see how well your resume matches.")

        if not st.session_state.current_resume_id:
            st.warning("Please upload and select a resume first!")
            return

        job_title = st.text_input("Job Title", placeholder="e.g., Senior ML Engineer at Sarvam AI")
        job_desc = st.text_area("Job Description", height=250, placeholder="Paste the full job description here...")

        if st.button("🔍 Analyze Match", type="primary"):
            if not job_desc.strip():
                st.error("Please enter a job description.")
            else:
                with st.spinner("Analyzing match..."):
                    try:
                        result = match_job(
                            st.session_state.user_id,
                            st.session_state.current_resume_id,
                            job_desc,
                            job_title or "Untitled",
                        )
                        st.markdown('<div class="info-box">', unsafe_allow_html=True)
                        st.metric("Semantic Similarity", f"{result['similarity_score']:.2f}")
                        st.markdown('</div>', unsafe_allow_html=True)
                        st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
                        st.markdown(result["analysis"])
                        st.markdown('</div>', unsafe_allow_html=True)
                    except AppError as e:
                        st.error(f"Error: {e.message}")

    # --- Page: Skill-Gap Analysis ---
    elif page == "📊 Skill-Gap Analysis":
        st.markdown('<p class="main-header">📊 Skill-Gap Analysis</p>', unsafe_allow_html=True)
        st.markdown("Identify exactly what skills you need to develop for your target job.")

        if not st.session_state.current_resume_id:
            st.warning("Please upload and select a resume first!")
            return

        job_desc = st.text_area("Target Job Description", height=250, placeholder="Paste the job description you're targeting...")

        if st.button("📊 Analyze Skill Gaps", type="primary"):
            if not job_desc.strip():
                st.error("Please enter a job description.")
            else:
                with st.spinner("Analyzing skill gaps..."):
                    try:
                        result = skill_gap_analysis(
                            st.session_state.user_id,
                            st.session_state.current_resume_id,
                            job_desc,
                        )
                        st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
                        st.markdown(result["analysis"])
                        st.markdown('</div>', unsafe_allow_html=True)
                    except AppError as e:
                        st.error(f"Error: {e.message}")

    # --- Page: Learning Roadmap ---
    elif page == "🗺️ Learning Roadmap":
        st.markdown('<p class="main-header">🗺️ Learning Roadmap</p>', unsafe_allow_html=True)
        st.markdown("Get a personalized, phased learning plan for your target role.")

        if not st.session_state.current_resume_id:
            st.warning("Please upload and select a resume first!")
            return

        target_role = st.text_input("Target Role", placeholder="e.g., ML Engineer, Data Scientist, Backend Developer")

        if st.button("🗺️ Generate Roadmap", type="primary"):
            if not target_role.strip():
                st.error("Please enter a target role.")
            else:
                with st.spinner("Generating your personalized roadmap..."):
                    try:
                        result = generate_roadmap(
                            st.session_state.user_id,
                            st.session_state.current_resume_id,
                            target_role,
                        )
                        st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
                        st.markdown(result["roadmap"])
                        st.markdown('</div>', unsafe_allow_html=True)
                    except AppError as e:
                        st.error(f"Error: {e.message}")

    # --- Page: Mock Interview ---
    elif page == "🎤 Mock Interview":
        st.markdown('<p class="main-header">🎤 Mock Interview Questions</p>', unsafe_allow_html=True)
        st.markdown("Practice with AI-generated interview questions tailored to your resume and target role.")

        if not st.session_state.current_resume_id:
            st.warning("Please upload and select a resume first!")
            return

        target_role = st.text_input("Target Role", placeholder="e.g., ML Engineer at Sarvam AI", key="int_role")
        num_q = st.slider("Number of questions", 5, 20, 10)

        if st.button("🎤 Generate Questions", type="primary"):
            if not target_role.strip():
                st.error("Please enter a target role.")
            else:
                with st.spinner("Generating interview questions..."):
                    try:
                        result = generate_interview_questions(
                            st.session_state.user_id,
                            st.session_state.current_resume_id,
                            target_role,
                            num_q,
                        )
                        st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
                        st.markdown(result["questions"])
                        st.markdown('</div>', unsafe_allow_html=True)
                    except AppError as e:
                        st.error(f"Error: {e.message}")

    # --- Page: Fit Score ---
    elif page == "📈 Fit Score":
        st.markdown('<p class="main-header">📈 Recruiter Fit Score</p>', unsafe_allow_html=True)
        st.markdown("Get a comprehensive recruiter-style evaluation of how well you fit the role.")

        if not st.session_state.current_resume_id:
            st.warning("Please upload and select a resume first!")
            return

        job_desc = st.text_area("Job Description", height=250, placeholder="Paste the job description...", key="fit_jd")

        if st.button("📈 Calculate Fit Score", type="primary"):
            if not job_desc.strip():
                st.error("Please enter a job description.")
            else:
                with st.spinner("Evaluating your fit... This may take a moment."):
                    try:
                        result = recruiter_fit_score(
                            st.session_state.user_id,
                            st.session_state.current_resume_id,
                            job_desc,
                        )
                        st.markdown('<div class="info-box">', unsafe_allow_html=True)
                        st.metric("Semantic Similarity Score", f"{result['semantic_similarity_score']:.1f}%")
                        st.markdown('</div>', unsafe_allow_html=True)
                        st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
                        st.markdown(result["evaluation"])
                        st.markdown('</div>', unsafe_allow_html=True)
                    except AppError as e:
                        st.error(f"Error: {e.message}")


# ============================================================================
# SECTION 11: DUAL-START ENTRY POINT
# ============================================================================

def start_fastapi():
    """Start FastAPI server in a daemon thread."""
    uvicorn.run(api, host="0.0.0.0", port=FASTAPI_PORT, log_level="info")


if __name__ == "__main__":
    import threading

    # Initialize database before starting anything
    init_db()

    # Check if we're being run by Streamlit or directly
    is_streamlit = "streamlit" in sys.modules or "_streamlit" in sys.modules

    # Start FastAPI in background thread when running via Streamlit
    #fastapi_thread = threading.Thread(target=start_fastapi, daemon=True)
    #fastapi_thread.start()
    #logger.info(f"FastAPI started on port {FASTAPI_PORT} (background thread)")
    if "fastapi_started" not in st.session_state:
        fastapi_thread = threading.Thread(target=start_fastapi, daemon=True)
        fastapi_thread.start()
        st.session_state.fastapi_started = True
        logger.info(f"FastAPI started on port {FASTAPI_PORT} (background thread)")

    # Launch Streamlit UI
    run_streamlit()
