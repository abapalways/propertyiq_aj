# PropertyIQ — Project Plan

## Project Summary
PropertyIQ is a real-estate agent assistant that helps first-time homebuyers 
go beyond simple listing search. It retrieves matching properties from a 
RAG-indexed corpus of listings and neighborhood data based on buyer 
criteria, calls tools to calculate real mortgage payments and pull 
comparable sales data (so numbers are computed, not guessed by the LLM), 
remembers each buyer's must-haves and previously rejected listings across 
sessions, and enforces fair-housing guardrails so it never filters or 
recommends based on protected characteristics.

## Persona
David and Elena Torres are a couple in their early 30s buying their first 
home. They're frustrated by listing sites that can't answer nuanced 
questions like "is this a good deal?" or "what's my real monthly payment 
with 10% down?" They want an assistant that remembers what they've already 
rejected, gives them straight numbers on affordability and comps, and helps 
them shortlist serious candidates faster.

## Tech Stack
- LLM: Ollama (qwen2.5), local, tool-calling capable
- Orchestration: LangChain
- Vector store: FAISS
- UI: Gradio
- Language: Python

## Roles
Solo build — I own all areas: RAG, tools/MCP, memory, guardrails, UI/observability

## Confirmation
- [x] Read requirements.md
- [x] Read tasks.md