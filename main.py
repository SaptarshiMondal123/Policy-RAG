import argparse
import json
import os
from rag_engine import RAGSystem

# ANSI escape codes for coloring terminal output (optional but looks pro)
GREEN = "\033[92m"
CYAN = "\033[96m"
RESET = "\033[0m"

def main():
    parser = argparse.ArgumentParser(description="Policy RAG Assistant (CLI)")
    parser.add_argument("--ingest", action="store_true", help="Ingest PDF documents from /data folder")
    parser.add_argument("--query", type=str, help="Question to ask the assistant")
    
    args = parser.parse_args()
    
    # Initialize System
    try:
        rag = RAGSystem()
    except ValueError as e:
        print(f"Error: {e}")
        return

    # --- MODE 1: INGESTION ---
    if args.ingest:
        print(f"{CYAN}🚀 Starting Ingestion Process...{RESET}")
        rag.ingest_data()
        print(f"{GREEN}✅ Ingestion Complete! Database is ready.{RESET}")
        return

    # --- MODE 2: QUERYING ---
    if args.query:
        print(f"\n{CYAN}🔍 Analyzing Question: {args.query}...{RESET}")
        
        # We use the 'advanced' prompt by default for the best experience
        response = rag.query(args.query, version="advanced")
        
        # The response is now a Python Dictionary (from JSON)
        answer = response.get('answer', "Error generating answer.")
        confidence = response.get('confidence', "Unknown")
        context_used = response.get('context_used', False)

        print("\n" + "="*40)
        print(f"🤖 {GREEN}ASSISTANT ANSWER:{RESET}")
        print("="*40)
        print(f"{answer}\n")
        
        print(f"📊 Confidence Score: {confidence}")
        print(f"📚 Context Referenced: {'Yes' if context_used else 'No'}")
        print("-" * 40)

        # Save to file log
        log_entry = {
            "query": args.query,
            "response": response
        }
        with open("query_history.json", "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
            
    else:
        # If user runs script without arguments, show help
        parser.print_help()

if __name__ == "__main__":
    main()