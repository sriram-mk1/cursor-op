import logging
from typing import List, Dict, Any

log = logging.getLogger("gateway")

class ContextOptimizer:
    """
    High-performance Context Truncation (Middle-Out).
    Keeps system, start, and end of conversation.
    """
    
    def __init__(self, max_chars: int = 40000):
        self.max_chars = max_chars
        self.keep_start = 5  # Keep first 5 messages
        self.keep_end = 10   # Keep last 10 messages
    
    def optimize(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Optimizes context by keeping the most important parts:
        1. System prompt
        2. Initial context (first few messages)
        3. Recent context (last few messages)
        """
        if len(messages) <= (self.keep_start + self.keep_end + 1):
            return messages
            
        system_msg = None
        other_msgs = []
        
        for msg in messages:
            if msg.get("role") == "system":
                if system_msg is None: system_msg = msg
            else:
                other_msgs.append(msg)
        
        if len(other_msgs) <= (self.keep_start + self.keep_end):
            return messages
            
        # Middle-out truncation
        start_msgs = other_msgs[:self.keep_start]
        end_msgs = other_msgs[-self.keep_end:]
        
        optimized = []
        if system_msg:
            optimized.append(system_msg)
        
        optimized.extend(start_msgs)
        
        # Add a small separator if we dropped messages
        dropped_count = len(other_msgs) - self.keep_start - self.keep_end
        if dropped_count > 0:
            optimized.append({
                "role": "user",
                "content": f"... [✂️ {dropped_count} messages truncated for context optimization] ..."
            })
            optimized.append({
                "role": "assistant",
                "content": "Understood. I'll continue based on the provided context."
            })
            
        optimized.extend(end_msgs)
        
        # Final safety check: character limit
        total_chars = sum(len(str(m.get("content", ""))) for m in optimized)
        if total_chars > self.max_chars:
            # If still too big, keep only system + last 5
            log.warning(f"⚠️ Context still too large ({total_chars} chars), aggressive truncation.")
            final = []
            if system_msg: final.append(system_msg)
            final.extend(other_msgs[-5:])
            return final
            
        log.info(f"✨ Truncated: {len(messages)} -> {len(optimized)} msgs ({dropped_count} dropped)")
        return optimized
