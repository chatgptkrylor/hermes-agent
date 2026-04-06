#!/usr/bin/env python3
"""
Hermes Agent with Native Ollama Support (Tool-Capable)

This script provides a Hermes-compatible agent that uses Ollama's native
/api/chat endpoint with full tool calling support.

Usage:
    python run_agent_ollama_tools.py --query "List files in current directory" --model "qwen3.5:latest"
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, List, Optional, Callable

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.ollama_adapter import OllamaClient, build_ollama_client, OLLAMA_DEFAULT_BASE_URL


# ── Tool Definitions ─────────────────────────────────────────────────────────

def get_basic_tools() -> List[Dict[str, Any]]:
    """Return basic tool definitions for Ollama.
    
    These are simplified tools that Ollama models can understand.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "execute_bash",
                "description": "Execute a bash command on the system",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The bash command to execute"
                        }
                    },
                    "required": ["command"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read the contents of a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "The path to the file to read"
                        }
                    },
                    "required": ["path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write content to a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "The path to the file to write"
                        },
                        "content": {
                            "type": "string",
                            "description": "The content to write to the file"
                        }
                    },
                    "required": ["path", "content"]
                }
            }
        }
    ]


def execute_tool(tool_name: str, arguments: Dict[str, Any]) -> str:
    """Execute a tool call.
    
    Args:
        tool_name: Name of the tool to execute
        arguments: Tool arguments
        
    Returns:
        Tool result as string
    """
    import subprocess
    
    if tool_name == "execute_bash":
        command = arguments.get("command", "")
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=60
            )
            return result.stdout or result.stderr or "(no output)"
        except subprocess.TimeoutExpired:
            return "Error: Command timed out"
        except Exception as e:
            return f"Error: {e}"
    
    elif tool_name == "read_file":
        path = arguments.get("path", "")
        try:
            with open(path, 'r') as f:
                return f.read()
        except Exception as e:
            return f"Error reading file: {e}"
    
    elif tool_name == "write_file":
        path = arguments.get("path", "")
        content = arguments.get("content", "")
        try:
            with open(path, 'w') as f:
                f.write(content)
            return f"Successfully wrote {len(content)} characters to {path}"
        except Exception as e:
            return f"Error writing file: {e}"
    
    else:
        return f"Unknown tool: {tool_name}"


def run_agent_with_tools(
    query: str,
    model: str = "qwen3.5:latest",
    base_url: str = OLLAMA_DEFAULT_BASE_URL,
    max_turns: int = 10,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    verbose: bool = False,
) -> str:
    """Run an agent loop with tool calling support.
    
    Args:
        query: User query
        model: Ollama model name
        base_url: Ollama server URL
        max_turns: Maximum conversation turns
        system_prompt: Optional system prompt
        temperature: Sampling temperature
        tools: Optional list of tools
        verbose: Print debug info
        
    Returns:
        Final assistant response
    """
    client = build_ollama_client(base_url)
    
    try:
        # Build initial messages
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": query})
        
        if verbose:
            print(f"🤖 Model: {model}")
            print(f"🌐 Base URL: {base_url}")
            print(f"🔧 Tools: {len(tools) if tools else 0}")
            print(f"📝 Query: {query[:100]}{'...' if len(query) > 100 else ''}")
            print()
        
        turn = 0
        while turn < max_turns:
            turn += 1
            
            if verbose:
                print(f"🔄 Turn {turn}/{max_turns}")
            
            # Call Ollama
            response = client.chat(
                model=model,
                messages=messages,
                tools=tools,
                temperature=temperature,
            )
            
            message = response.get("message", {})
            content = message.get("content", "")
            tool_calls = message.get("tool_calls")
            
            if verbose:
                if content:
                    print(f"📤 Content: {content[:200]}{'...' if len(content) > 200 else ''}")
                if tool_calls:
                    print(f"🔧 Tool calls: {len(tool_calls)}")
            
            # Check if we're done
            if not tool_calls:
                # Final response
                if verbose:
                    print(f"\n✅ Completed in {turn} turns")
                    print(f"📊 Tokens: prompt={response.get('prompt_eval_count', 0)}, "
                          f"completion={response.get('eval_count', 0)}")
                return content
            
            # Add assistant message with tool calls
            messages.append({
                "role": "assistant",
                "content": content,
                "tool_calls": tool_calls
            })
            
            # Execute tool calls
            for tc in tool_calls:
                func = tc.get("function", {})
                tool_name = func.get("name", "")
                arguments = func.get("arguments", {})
                
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {}
                
                if verbose:
                    print(f"  ⚙️ {tool_name}({arguments})")
                
                result = execute_tool(tool_name, arguments)
                
                if verbose:
                    print(f"  📋 Result: {result[:100]}{'...' if len(result) > 100 else ''}")
                
                # Add tool result
                messages.append({
                    "role": "tool",
                    "content": result,
                    "tool_call_id": tc.get("id", "")
                })
        
        # Max turns reached
        if verbose:
            print(f"\n⚠️ Max turns ({max_turns}) reached")
        return content or "(no response)"
        
    finally:
        client.close()


def interactive_mode(
    model: str = "qwen3.5:latest",
    base_url: str = OLLAMA_DEFAULT_BASE_URL,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    verbose: bool = False,
):
    """Run in interactive mode.
    
    Args:
        model: Ollama model name
        base_url: Ollama server URL
        system_prompt: Optional system prompt
        temperature: Sampling temperature
        tools: Optional list of tools
        verbose: Print debug info
    """
    client = build_ollama_client(base_url)
    
    print("=" * 60)
    print(f"🤖 Ollama Agent (native /api/chat with tools)")
    print(f"   Model: {model}")
    print(f"   Base URL: {base_url}")
    print(f"   Tools: {len(tools) if tools else 0}")
    print("=" * 60)
    print("Commands: /exit, /quit, /clear, /help")
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
            
            # Commands
            if query.lower() in ("/exit", "/quit", "exit", "quit"):
                print("👋 Goodbye!")
                break
            
            if query.lower() == "/clear":
                conversation = [{"role": "system", "content": system_prompt}] if system_prompt else []
                print("🧹 Conversation cleared")
                continue
            
            if query.lower() == "/help":
                print("Commands: /exit, /quit, /clear, /help")
                continue
            
            # Run query with tools
            conversation.append({"role": "user", "content": query})
            
            try:
                response = client.chat(
                    model=model,
                    messages=conversation,
                    tools=tools,
                    temperature=temperature,
                )
                
                message = response.get("message", {})
                content = message.get("content", "")
                tool_calls = message.get("tool_calls")
                
                # Handle tool calls
                while tool_calls:
                    # Add assistant message
                    conversation.append({
                        "role": "assistant",
                        "content": content,
                        "tool_calls": tool_calls
                    })
                    
                    # Execute tools
                    for tc in tool_calls:
                        func = tc.get("function", {})
                        tool_name = func.get("name", "")
                        arguments = func.get("arguments", {})
                        
                        if isinstance(arguments, str):
                            try:
                                arguments = json.loads(arguments)
                            except:
                                arguments = {}
                        
                        result = execute_tool(tool_name, arguments)
                        
                        if verbose:
                            print(f"  ⚙️ {tool_name}({arguments})")
                            print(f"  📋 {result[:100]}")
                        
                        conversation.append({
                            "role": "tool",
                            "content": result,
                            "tool_call_id": tc.get("id", "")
                        })
                    
                    # Get next response
                    response = client.chat(
                        model=model,
                        messages=conversation,
                        tools=tools,
                        temperature=temperature,
                    )
                    
                    message = response.get("message", {})
                    content = message.get("content", "")
                    tool_calls = message.get("tool_calls")
                
                if content:
                    conversation.append({"role": "assistant", "content": content})
                    print(f"\n{content}\n")
                else:
                    print("(no response)\n")
                
            except Exception as e:
                print(f"❌ Error: {e}")
                # Remove the failed message
                if conversation and conversation[-1].get("role") == "user":
                    conversation.pop()
                
    finally:
        client.close()


def main():
    parser = argparse.ArgumentParser(
        description="Run Hermes Agent with native Ollama support (tool-capable)"
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
        default=10,
        help="Maximum conversation turns (default: 10)"
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
        "--no-tools",
        action="store_true",
        help="Disable tools"
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
    
    tools = None if args.no_tools else get_basic_tools()
    
    if args.interactive or not args.query:
        interactive_mode(
            model=args.model,
            base_url=args.base_url,
            system_prompt=args.system,
            temperature=args.temperature,
            tools=tools,
            verbose=args.verbose,
        )
    else:
        result = run_agent_with_tools(
            query=args.query,
            model=args.model,
            base_url=args.base_url,
            max_turns=args.max_turns,
            system_prompt=args.system,
            temperature=args.temperature,
            tools=tools,
            verbose=args.verbose,
        )
        print(result)


if __name__ == "__main__":
    main()