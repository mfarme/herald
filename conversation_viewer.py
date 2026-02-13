#!/usr/bin/env python3
"""
Conversation Viewer - Displays JSONL conversations in a web interface
"""
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import os

JSONL_FILE = "data/combined_dataset.jsonl"

def load_conversations():
    """Load and parse conversations from JSONL file"""
    conversations = []
    current_conversation = None
    
    with open(JSONL_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            
            if data.get('type') == 'metadata':
                # Save previous conversation if exists
                if current_conversation:
                    conversations.append(current_conversation)
                
                # Start new conversation
                current_conversation = {
                    'metadata': data,
                    'messages': []
                }
            elif current_conversation:
                # Add message to current conversation
                current_conversation['messages'].append(data)
        
        # Don't forget the last conversation
        if current_conversation:
            conversations.append(current_conversation)
    
    return conversations

def generate_html(conversations, selected_idx=None):
    """Generate HTML page with conversation list and viewer"""
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Conversation Viewer</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            display: flex;
            height: 100vh;
            background: #f5f5f5;
        }}
        
        .sidebar {{
            width: 350px;
            background: white;
            border-right: 1px solid #ddd;
            overflow-y: auto;
        }}
        
        .sidebar-header {{
            padding: 20px;
            background: #2c3e50;
            color: white;
            font-size: 18px;
            font-weight: 600;
        }}
        
        .conversation-list {{
            list-style: none;
        }}
        
        .conversation-item {{
            padding: 15px 20px;
            border-bottom: 1px solid #eee;
            cursor: pointer;
            transition: background 0.2s;
        }}
        
        .conversation-item:hover {{
            background: #f8f9fa;
        }}
        
        .conversation-item.active {{
            background: #e3f2fd;
            border-left: 4px solid #2196f3;
        }}
        
        .conv-title {{
            font-weight: 600;
            margin-bottom: 5px;
            color: #333;
        }}
        
        .conv-meta {{
            font-size: 12px;
            color: #666;
        }}
        
        .severity-badge {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
            margin-left: 5px;
        }}
        
        .severity-0 {{ background: #c8e6c9; color: #2e7d32; }}
        .severity-1 {{ background: #fff9c4; color: #f57f17; }}
        .severity-2 {{ background: #ffcdd2; color: #c62828; }}
        
        .main-content {{
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }}
        
        .conversation-header {{
            padding: 20px 30px;
            background: white;
            border-bottom: 1px solid #ddd;
        }}
        
        .metadata-box {{
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 10px;
        }}
        
        .metadata-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin-bottom: 8px;
        }}
        
        .metadata-item {{
            font-size: 14px;
        }}
        
        .metadata-label {{
            font-weight: 600;
            color: #555;
        }}
        
        .metadata-value {{
            color: #333;
        }}
        
        .chat-container {{
            flex: 1;
            overflow-y: auto;
            padding: 30px;
            background: #f5f5f5;
        }}
        
        .message {{
            max-width: 70%;
            margin-bottom: 20px;
            animation: fadeIn 0.3s;
        }}
        
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        .message.user {{
            margin-left: auto;
        }}
        
        .message-role {{
            font-size: 12px;
            font-weight: 600;
            margin-bottom: 5px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .message.user .message-role {{
            color: #1976d2;
            text-align: right;
        }}
        
        .message.assistant .message-role {{
            color: #388e3c;
        }}
        
        .message-bubble {{
            padding: 15px 18px;
            border-radius: 18px;
            line-height: 1.5;
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
        
        .message.user .message-bubble {{
            background: #2196f3;
            color: white;
            border-bottom-right-radius: 4px;
        }}
        
        .message.assistant .message-bubble {{
            background: white;
            color: #333;
            border: 1px solid #e0e0e0;
            border-bottom-left-radius: 4px;
        }}
        
        .empty-state {{
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100%;
            color: #999;
            font-size: 18px;
        }}
    </style>
</head>
<body>
    <div class="sidebar">
        <div class="sidebar-header">
            Conversations ({total_count})
        </div>
        <ul class="conversation-list">
            {conversation_list}
        </ul>
    </div>
    
    <div class="main-content">
        {main_content}
    </div>
</body>
</html>"""
    
    # Generate conversation list items
    list_items = []
    for idx, conv in enumerate(conversations):
        meta = conv['metadata']
        active_class = 'active' if idx == selected_idx else ''
        severity_class = f"severity-{meta['severity_level']}"
        
        list_items.append(f"""
            <li class="conversation-item {active_class}" onclick="window.location.href='?conv={idx}'">
                <div class="conv-title">
                    {meta['persona_name']}
                    <span class="severity-badge {severity_class}">Level {meta['severity_level']}</span>
                </div>
                <div class="conv-meta">
                    {meta['conversation_type'].replace('_', ' ').title()} | {meta['turns']} turns
                </div>
            </li>
        """)
    
    # Generate main content
    if selected_idx is not None and 0 <= selected_idx < len(conversations):
        conv = conversations[selected_idx]
        meta = conv['metadata']
        
        # Metadata box
        metadata_html = f"""
        <div class="conversation-header">
            <div class="metadata-box">
                <div class="metadata-row">
                    <div class="metadata-item">
                        <span class="metadata-label">Persona:</span>
                        <span class="metadata-value">{meta['persona_name']}</span>
                    </div>
                    <div class="metadata-item">
                        <span class="metadata-label">Type:</span>
                        <span class="metadata-value">{meta['conversation_type'].replace('_', ' ').title()}</span>
                    </div>
                    <div class="metadata-item">
                        <span class="metadata-label">Severity Level:</span>
                        <span class="metadata-value">{meta['severity_level']}</span>
                    </div>
                    <div class="metadata-item">
                        <span class="metadata-label">Turns:</span>
                        <span class="metadata-value">{meta['turns']}</span>
                    </div>
                </div>
            </div>
        </div>
        """
        
        # Messages
        messages_html = '<div class="chat-container">'
        for msg in conv['messages']:
            role = msg['role']
            content = msg['content']
            messages_html += f"""
            <div class="message {role}">
                <div class="message-role">{role}</div>
                <div class="message-bubble">{content}</div>
            </div>
            """
        messages_html += '</div>'
        
        main_content = metadata_html + messages_html
    else:
        main_content = '<div class="empty-state">← Select a conversation to view</div>'
    
    return html.format(
        total_count=len(conversations),
        conversation_list=''.join(list_items),
        main_content=main_content
    )

class ConversationHandler(BaseHTTPRequestHandler):
    conversations = None
    
    def do_GET(self):
        """Handle GET requests"""
        if ConversationHandler.conversations is None:
            ConversationHandler.conversations = load_conversations()
        
        parsed_url = urlparse(self.path)
        query_params = parse_qs(parsed_url.query)
        
        # Get selected conversation index
        selected_idx = None
        if 'conv' in query_params:
            try:
                selected_idx = int(query_params['conv'][0])
            except (ValueError, IndexError):
                pass
        
        # Generate and send HTML
        html = generate_html(ConversationHandler.conversations, selected_idx)
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))
    
    def log_message(self, format, *args):
        """Custom log format"""
        print(f"[{self.log_date_time_string()}] {format % args}")

def main():
    """Start the server"""
    port = 8000
    server_address = ('', port)
    
    print(f"Loading conversations from {JSONL_FILE}...")
    print(f"\nStarting server on http://localhost:{port}")
    print("Press Ctrl+C to stop the server\n")
    
    httpd = HTTPServer(server_address, ConversationHandler)
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\nShutting down server...")
        httpd.shutdown()

if __name__ == "__main__":
    main()
