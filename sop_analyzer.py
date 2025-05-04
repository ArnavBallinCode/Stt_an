import re
from datetime import datetime

class SOPAnalyzer:
    def __init__(self, sop_file="agent_sop.txt"):
        self.sop_rules = self._load_sop(sop_file)
        self.analysis_results = {
            "greeting_protocol": {"score": 0, "total": 0, "details": []},
            "problem_identification": {"score": 0, "total": 0, "details": []},
            "solution_steps": {"score": 0, "total": 0, "details": []},
            "closing_protocol": {"score": 0, "total": 0, "details": []},
            "prohibited_phrases": {"score": 0, "total": 0, "details": []},
            "information_collection": {"score": 0, "total": 0, "details": []}
        }
    
    def _load_sop(self, sop_file):
        """Load and parse the SOP file"""
        with open(sop_file, 'r') as f:
            content = f.read()
        
        rules = {}
        current_section = None
        
        for line in content.split('\n'):
            if line.startswith('## '):
                current_section = line[3:].lower().replace(' ', '_')
                rules[current_section] = []
            elif line.startswith('- '):
                if current_section:
                    rules[current_section].append(line[2:])
        
        return rules
    
    def analyze_message(self, message, speaker):
        """Analyze a single message against SOP rules"""
        if speaker != "Speaker 2":  # Only analyze agent messages
            return
        
        message_lower = message.lower()
        
        # Greeting Protocol
        if any(greeting in message_lower for greeting in ["hello", "hi", "good morning", "good afternoon"]):
            self.analysis_results["greeting_protocol"]["score"] += 1
            self.analysis_results["greeting_protocol"]["details"].append(
                f"✓ Used appropriate greeting at {datetime.now().strftime('%H:%M:%S')}"
            )
        self.analysis_results["greeting_protocol"]["total"] += 1
        
        # Problem Identification
        if any(phrase in message_lower for phrase in ["what seems to be", "what's the issue", "how can i help"]):
            self.analysis_results["problem_identification"]["score"] += 1
            self.analysis_results["problem_identification"]["details"].append(
                f"✓ Asked about the problem at {datetime.now().strftime('%H:%M:%S')}"
            )
        self.analysis_results["problem_identification"]["total"] += 1
        
        # Solution Steps
        if any(phrase in message_lower for phrase in ["next step", "please provide", "upload", "send"]):
            self.analysis_results["solution_steps"]["score"] += 1
            self.analysis_results["solution_steps"]["details"].append(
                f"✓ Provided clear next steps at {datetime.now().strftime('%H:%M:%S')}"
            )
        self.analysis_results["solution_steps"]["total"] += 1
        
        # Closing Protocol
        if any(phrase in message_lower for phrase in ["thank you", "anything else", "is there anything else"]):
            self.analysis_results["closing_protocol"]["score"] += 1
            self.analysis_results["closing_protocol"]["details"].append(
                f"✓ Used appropriate closing at {datetime.now().strftime('%H:%M:%S')}"
            )
        self.analysis_results["closing_protocol"]["total"] += 1
        
        # Prohibited Phrases
        for phrase in self.sop_rules["prohibited_phrases"]:
            if phrase.lower() in message_lower:
                self.analysis_results["prohibited_phrases"]["score"] -= 1
                self.analysis_results["prohibited_phrases"]["details"].append(
                    f"✗ Used prohibited phrase '{phrase}' at {datetime.now().strftime('%H:%M:%S')}"
                )
        self.analysis_results["prohibited_phrases"]["total"] += 1
        
        # Information Collection
        if any(phrase in message_lower for phrase in ["order number", "product details", "contact information"]):
            self.analysis_results["information_collection"]["score"] += 1
            self.analysis_results["information_collection"]["details"].append(
                f"✓ Requested required information at {datetime.now().strftime('%H:%M:%S')}"
            )
        self.analysis_results["information_collection"]["total"] += 1
    
    def get_analysis_report(self):
        """Generate a detailed analysis report"""
        report = []
        report.append("\n=== Agent SOP Compliance Analysis ===\n")
        
        for section, results in self.analysis_results.items():
            if results["total"] > 0:
                score = (results["score"] / results["total"]) * 100
                report.append(f"\n{section.replace('_', ' ').title()}:")
                report.append(f"Compliance Score: {score:.1f}%")
                report.append("Details:")
                for detail in results["details"]:
                    report.append(f"  {detail}")
        
        return "\n".join(report)

def main():
    # Example usage
    analyzer = SOPAnalyzer()
    
    # Example conversation
    conversation = [
        ("Speaker 2", "Hello, how can I help you today?"),
        ("Speaker 1", "I have an issue with my order"),
        ("Speaker 2", "I'm sorry to hear that. Could you please provide your order number?"),
        ("Speaker 1", "It's #12345"),
        ("Speaker 2", "Thank you. Could you describe the issue in detail?"),
        ("Speaker 1", "The product arrived damaged"),
        ("Speaker 2", "I apologize for the inconvenience. Could you please upload some photos of the damage?"),
        ("Speaker 1", "Sure, I'll do that"),
        ("Speaker 2", "Thank you for providing the information. We'll process your request within 24 hours. Is there anything else I can help you with?")
    ]
    
    # Analyze each message
    for speaker, message in conversation:
        analyzer.analyze_message(message, speaker)
    
    # Print analysis report
    print(analyzer.get_analysis_report())

if __name__ == "__main__":
    main() 