from openai import OpenAI
import time
import os
from typing import Dict, List

class ChatSummarizer:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def generate_summary(self, messages: List[str], keywords: List[str]) -> str:
        prompt = (
            "Analiza y resume la siguiente conversación de WhatsApp.\n\n"
            f"Palabras clave identificadas: {', '.join(keywords)}\n\n"
            "Mensajes principales:\n"
            + '\n'.join(messages[:50]) +
            "\n\nPor favor, proporciona:\n"
            "1. Un resumen conciso de los temas principales\n"
            "2. Puntos clave de la discusión\n"
            "3. Conclusiones o decisiones importantes"
        )

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo-16k",
                messages=[
                    {"role": "system", "content": "Eres un experto en análisis y resumen de conversaciones."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating summary: {str(e)}"

    def generate_summaries(self, results_dir: str) -> Dict[int, str]:
        summaries = {}
        
        for file in os.listdir(results_dir):
            if file.startswith("tema_") and file.endswith(".txt"):
                topic_num = int(file.split("_")[1].split(".")[0])
                file_path = os.path.join(results_dir, file)
                
                messages, keywords = self._extract_content(file_path)
                summary = self.generate_summary(messages, keywords)
                summaries[topic_num] = summary
                
                time.sleep(1)  # Rate limiting
                
        self._save_summaries(results_dir, summaries)
        return summaries

    def _extract_content(self, file_path: str) -> tuple:
        messages = []
        keywords = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if "PALABRAS CLAVE DEL TEMA:" in content:
                keywords_section = content.split("PALABRAS CLAVE DEL TEMA:")[1].split("\n")[1]
                keywords = [k.strip() for k in keywords_section.split(",")]
            
            if "MENSAJES DEL TEMA:" in content:
                messages_section = content.split("MENSAJES DEL TEMA:")[1]
                messages = [m.strip() for m in messages_section.split("\n") 
                          if m.strip() and not m.strip().startswith("=") 
                          and not m.strip().startswith("-")]
                
        return messages, keywords

    def _save_summaries(self, results_dir: str, summaries: Dict[int, str]):
        summaries_dir = os.path.join(results_dir, "resumenes_gpt")
        os.makedirs(summaries_dir, exist_ok=True)
        
        # Save individual summaries
        for topic_num, summary in summaries.items():
            file_path = os.path.join(summaries_dir, f"resumen_gpt_tema_{topic_num}.txt")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"RESUMEN - TEMA {topic_num}\n")
                f.write("="*50 + "\n\n")
                f.write(summary)
        
        # Save general summary
        general_summary_path = os.path.join(summaries_dir, "resumen_general_gpt.txt")
        with open(general_summary_path, 'w', encoding='utf-8') as f:
            f.write("RESUMEN GENERAL DE TODOS LOS TEMAS\n")
            f.write("="*50 + "\n\n")
            for topic_num, summary in sorted(summaries.items()):
                f.write(f"\nTEMA {topic_num}\n")
                f.write("-"*20 + "\n")
                f.write(summary + "\n")