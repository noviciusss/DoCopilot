import re
from typing import Tuple 

class RagGuardrails:
    """Firsly trying basic implamentataion then will replace it with guardrailsai lib pata to chale ho kya rha hai"""
    
    INJECTION_PATTERNS = [
        r"ignore (previous|all|prior) instructions",
        r"forget (everything|your rules)",
        r"you are now",
        r"pretend to be",
        r"act as if",
        r"system prompt"
    ]
    
    PII_PATTERNS = {
        "credit_card": r"\b(?:\d[ -]*?){13,16}\b",
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "phone": r"\b(?:\+91[\s-]?|0)?[6-9]\d{9}\b"
    }
    
    @classmethod
    def check_input(cls, query: str)-> Tuple[bool,str]:
        """validate user input before processing
        returns (is_safe,message) 
        """
        query_lower = query.lower()
        
        for pattern in cls.INJECTION_PATTERNS:
            if re.search(pattern , query_lower):
                return False, "Potential prompt injection detected."
        
        ## length model ke aukat ke bahar na jaye    
        if len(query)> 2000:
            return False,"Input too long , pls keep it under 2000 kripa kar ke thoda chhota soche :)"
        
        if len(query.strip())<3:
            return False, "Query too short , Credits waste mat karo bhai :)"
        
        return True, "OK"
    
    @classmethod
    def redact_pii(cls,text:str)->str:
        """Redant PII from the given text
        """        
        for pii_type, pattern in cls.PII_PATTERNS.items():
            text = re.sub(pattern ,f"[REDACTED {pii_type.upper()}]", text)
        return text
    
    @classmethod
    def check_output(cls,output:str,sources:list)-> Tuple[bool,str]:
        """validate llm output before output 
        return (is_safe,message)"""
        
        cleaned = cls.redact_pii(output)
        
        refusal_phrases = [
            "i'm sorry",
            "i cannot",
            "i'm unable to",
            "i do not have the capability",
            "as an ai language model"
        
        ]
        response_lower = cleaned.lower()
        
        has_refusal = any(phrase in response_lower for phrase in refusal_phrases)
        if not sources  and not has_refusal:
            cleaned = cleaned + "\n\n *NOte : so ye jo answer given hai wo given sources pe based nahi hai , kripa kar ke verify kar lein*"
        
        return True, cleaned
    
    ##method to acha hai but words limit hai , so thats a problem 
    @classmethod 
    def check_relevance(cls,query:str,document_content:str)-> bool:
        """ basic check if query is relavent hai with document content"""
        query_words = set(query.lower().split())
        doc_words = set(document_content.lower().split()[:500])
        
        overlap = len(query_words & doc_words)
        return overlap >= 3