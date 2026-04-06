#!/usr/bin/env python3
"""
Hermes Agent with Native Ollama Support

This script wraps Hermes Agent to use Ollama's native /api/chat endpoint
instead of the OpenAI-compatible endpoint. This bypasses the tool definition
issue that causes Ollama to reject requests.

Usage:
    python run_agent_ollama.py --query "What is 2+2?" --model "qwen3.5:latest"
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.ollama_adapter import OllamaClient, build_ollama_client, OLLAMA_DEFAULT_BASE_URL


def run_ollama_query(
    query: str,
    model: str = "qwen3.5:latest",
    base_url: str = OLLAMA_DEFAULT_BASE_URL,
    max_turns: int = 5,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    verbose: bool = False,
) -> str:
    """Run a query using Ollama native API.
    
    Args:
        query: User query
        model: Ollama model name
        base_url: Ollama server URL
        max_turns: Maximum conversation turns (for future tool support)
        system_prompt: Optional system prompt
        temperature: Sampling temperature
        verbose: Print debug info
        
    Returns:
        Assistant response
    """
    client = build_ollama_client(base_url)
    
    try:
        # Build messages
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": query})
        
        if verbose:
            print(f"🤖 Model: {model}")
            print(f"🌐 Base URL: {base_url}")
            print(f"📝 Query: {query[:100]}{'...' if len(query) > 100 else ''}")
            print()
        
        # Call Ollama
        response = client.chat(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        
        # Extract response
        message = response.get("message", {})
        content = message.get("content", "")
        
        if verbose:
            print(f"📤 Response: {content[:200]}{'...' if len(content) > 200 else ''}")
            print()
            print(f"📊 Tokens: prompt={response.get('prompt_eval_count', 0)}, "
                  f"completion={response.get('eval_count', 0)}")
        
        return content
        
    finally:
        client.close()


def interactive_mode(
    model: str = "qwen3.5:latest",
    base_url: str = OLLAMA_DEFAULT_BASE_URL,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
):
    """Run in interactive mode.
    
    Args:
        model: Ollama model name
        base_url: Ollama server URL
        system_prompt: Optional system prompt
        temperature: Sampling temperature
    """
    client = build_ollama_client(base_url)
    
    print("=" * 60)
    print(f"🤖 Ollama Agent (native /api/chat)")
    print(f"   Model: {model}")
    print(f"   Base URL: {base_url}")
    print("=" * 60)
    print()
    
    conversation: List[Dict[str, Any]] = []
    
    if system_prompt:
        conversation.append({"role": "system", "content": system_prompt})
    
    try:
        while True:
            try:
                query = input("❯ ").strip()
            except EOFError:
                print()
                break
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            
            if not query:
                continue
            
            if query.lower() in ("/exit", "/quit", "exit", "quit"):
                print("👋 Goodbye!")
                break
            
            conversation.append({"role": "user", "content": query})
            
            try:
                response = client.chat(
                    model=model,
                    messages=conversation,
                    temperature=temperature,
                )
                
                message = response.get("message", {})
                content = message.get("content", "")
                
                if content:
                    conversation.append({"role": "assistant", "content": content})
                    print(f"\n{content}\n")
                else:
                    print("(no response)")
                
            except Exception as e:
                print(f"❌ Error: {e}")
                # Remove the failed message
                conversation.pop()
                
    finally:
        client.close()


def main():
    parser = argparse.ArgumentParser(
        description="Run Hermes Agent with native Ollama support"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Query to run (non-interactive mode)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="qwen3.5:latest",
        help="Ollama model name (default: qwen3.5:latest)"
    )
    parser.add_argument(
        "--base-url", "-b",
        type=str,
        default=OLLAMA_DEFAULT_BASE_URL,
        help=f"Ollama server URL (default: {OLLAMA_DEFAULT_BASE_URL})"
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=5,
        help="Maximum conversation turns (default: 5)"
    )
    parser.add_argument(
        "--system", "-s",
        type=str,
        default=None,
        help="System prompt"
    )
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=None,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Interactive mode"
    )
    
    args = parser.parse_args()
    
    if args.interactive or not args.query:
        interactive_mode(
            model=args.model,
            base_url=args.base_url,
            system_prompt=args.system,
            temperature=args.temperature,
        )
    else:
        result = run_ollama_query(
            query=args.query,
            model=args.model,
            base_url=args.base_url,
            max_turns=args.max_turns,
            system_prompt=args.system,
            temperature=args.temperature,
            verbose=args.verbose,
        )
        print(result)


if __name__ == "__main__":
    main()