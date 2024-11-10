import pandas as pd
import numpy as np
import re
import os
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from nltk.stem import WordNetLemmatizer
import nltk
import zipfile
import tempfile
import shutil


class ChatAnalyzer:
    def __init__(self):
        # Initialize NLTK resources
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)

    def clean_message(self, text):
        if pd.isna(text):
            return ""

        text_clean = text.lower()
        patterns = [
            r"[\U0001F300-\U0001F9FF]",
            r"\b(?:xd+|:v|v:|umu|uwu|:\'v|:\'\'v)\b",
            r"http[s]?://\S+",
            r"stk-\d+-wa\d+\.webp",
            r"img-\d+-wa\d+\.jpg",
            r"<se editó este mensaje\.>",
            r"<multimedia omitido>",
            r"\bx\d+\b",
            r"\(archivo adjunto\)",
            r"zzz+",
            r"se eliminó este mensaje\.",
            r"\b(enlace|unir|unió|unido)\b",
            r"(añadió|añadiste)\b",
        ]

        for pattern in patterns:
            text_clean = re.sub(pattern, "", text_clean)

        return text_clean.strip()

    def load_chat(self, file_path):
        try:
            messages = []
        
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    line = line.strip()
                    if not line:
                        continue
                
                    # Patrón actualizado para coincidir con el formato:
                    # "DD/MM/YY, HH:mm - Usuario: Mensaje"
                    pattern = r'(\d{2}/\d{2}/\d{2}),\s(\d{2}:\d{2})\s-\s(\d+):\s(.+)'
                    match = re.match(pattern, line)

                    if match:
                        date, time, user, message = match.groups()
                        
                        if message:
                            # Limpieza del mensaje
                            clean_message = self.clean_message(message)
                            
                            if clean_message:
                                messages.append({
                                    'fecha': date,
                                    'hora': time,
                                    'usuario': user.strip(),
                                    'mensaje_original': message,
                                    'mensaje_limpio': clean_message
                                })

            if not messages:
                print("Contenido del archivo:")
                with open(file_path, 'r', encoding='utf-8') as file:
                    print(file.read())
                raise ValueError("No se encontraron mensajes válidos en el archivo")

            return pd.DataFrame(messages)
        
        except Exception as e:
            print(f"Error al procesar el archivo: {str(e)}")
            print("Primeras líneas del archivo:")
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    print(file.read()[:500])
            except:
                print("No se pudo leer el archivo")
            raise Exception(f"Error al cargar el chat: {str(e)}")

    def cluster_messages(self, df, method="lda", n_groups=5):
        try:
            if len(df) < n_groups:
                raise ValueError(
                    f"No hay suficientes mensajes para crear {n_groups} grupos"
                )

            vectorizer = TfidfVectorizer(
                min_df=2,
                max_df=0.90,
                ngram_range=(1, 2),
                stop_words=self._get_stop_words(),
            )

            X = vectorizer.fit_transform(df["mensaje_limpio"])

            if method == "kmeans":
                model = KMeans(n_clusters=n_groups, random_state=42, n_init=10)
                clusters = model.fit_predict(X)
                keywords = self._get_kmeans_keywords(model, vectorizer)
            else:  # lda
                model = LatentDirichletAllocation(
                    n_components=n_groups, random_state=42
                )
                topic_distribution = model.fit_transform(X)
                clusters = topic_distribution.argmax(axis=1)
                keywords = self._get_lda_keywords(model, vectorizer)

            df["cluster"] = clusters
            return df, keywords
        except Exception as e:
            raise Exception(f"Error en el clustering de mensajes: {str(e)}")

    def save_results(self, df, keywords, temp_dir):
        try:
            os.makedirs(temp_dir, exist_ok=True)

            # Los archivos cluster de los mensajes
            for cluster in df["cluster"].unique():
                df_cluster = df[df["cluster"] == cluster]
                self._save_cluster_file(
                    df_cluster, cluster, keywords[cluster], temp_dir
                )

            # Se guardara el archivo resumen
            self._save_summary_file(df, keywords, temp_dir)

        except Exception as e:
            raise Exception(f"Error al guardar resultados: {str(e)}")

    def analyze(self, file_path, method="lda"):
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"No se encontró el archivo: {file_path}")

            df = self.load_chat(file_path)
            if len(df) == 0:
                raise ValueError("No se encontraron mensajes válidos en el chat")

            # Crear un directorio temporal para los archivos
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_dir = tempfile.mkdtemp()

            try:
                # Procesar y guardar los resultados en el directorio temporal
                df_processed, keywords = self.cluster_messages(df, method)
                self.save_results(df_processed, keywords, temp_dir)

                # Crear el archivo ZIP
                zip_filename = f"resultados_chat_{timestamp}.zip"
                zip_path = os.path.join(os.getcwd(), zip_filename)

                with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                    # Agregar todos los archivos del directorio temporal al ZIP
                    for root, _, files in os.walk(temp_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, temp_dir)
                            zipf.write(file_path, arcname)

                return zip_path

            finally:
                # Limpiar el directorio temporal
                shutil.rmtree(temp_dir, ignore_errors=True)

                # Limpiar el archivo temporal de entrada
                if file_path.startswith("temp_"):
                    try:
                        os.remove(file_path)
                    except:
                        pass

        except Exception as e:
            raise Exception(f"Error en el análisis: {str(e)}")

    def _get_stop_words(self):
        return [
            "de",
            "la",
            "que",
            "el",
            "en",
            "y",
            "a",
            "los",
            "se",
            "del",
            "las",
            "un",
            "por",
            "con",
            "una",
            "su",
            "para",
            "es",
            "al",
            "lo",
            "como",
            "mas",
            "pero",
            "sus",
            "le",
            "ya",
            "o",
            "este",
            "si",
            "porque",
            "muy",
            "sin",
            "sobre",
            "mi",
            "hay",
            "bien",
            "cuando",
            "ahora",
            "esta",
            "asi",
            "nos",
            "ni",
            "ese",
            "eso",
            "esto",
            "etc",
            "otro",
            "tras",
        ]

    def _get_kmeans_keywords(self, model, vectorizer):
        try:
            keywords = {}
            feature_names = vectorizer.get_feature_names_out()
            for i, center in enumerate(model.cluster_centers_):
                top_indices = center.argsort()[-20:][::-1]
                keywords[i] = [feature_names[idx] for idx in top_indices]
            return keywords
        except Exception as e:
            raise Exception(f"Error al obtener keywords de KMeans: {str(e)}")

    def _get_lda_keywords(self, model, vectorizer):
        try:
            keywords = {}
            feature_names = vectorizer.get_feature_names_out()
            for i, topic in enumerate(model.components_):
                top_indices = topic.argsort()[-20:][::-1]
                keywords[i] = [feature_names[idx] for idx in top_indices]
            return keywords
        except Exception as e:
            raise Exception(f"Error al obtener keywords de LDA: {str(e)}")

    def _save_cluster_file(self, df_cluster, cluster_num, keywords, directory):
        try:
            file_path = os.path.join(directory, f"tema_{cluster_num+1}.txt")

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(f"TEMA {cluster_num+1}\n")
                f.write("=" * 50 + "\n\n")

                # Escribir las palabras claves
                f.write("PALABRAS CLAVE DEL TEMA:\n")
                f.write(", ".join(keywords) + "\n\n")

                # Escribir estadisticas
                f.write(f"Total mensajes: {len(df_cluster)}\n")
                f.write(f"Participantes: {df_cluster['usuario'].nunique()}\n")
                f.write(
                    f"Período: {df_cluster['fecha'].min()} a {df_cluster['fecha'].max()}\n\n"
                )

                # Escribir mensajes
                f.write("MENSAJES DEL TEMA:\n")
                f.write("-" * 50 + "\n\n")

                for fecha in sorted(df_cluster["fecha"].unique()):
                    mensajes_fecha = df_cluster[df_cluster["fecha"] == fecha]
                    f.write(f"\n[{fecha}]\n")
                    for _, row in mensajes_fecha.iterrows():
                        f.write(
                            f"{row['hora']} - {row['usuario']}: {row['mensaje_original']}\n"
                        )
        except Exception as e:
            raise Exception(f"Error al guardar archivo del cluster: {str(e)}")

    def _save_summary_file(self, df, keywords, directory):
        try:
            file_path = os.path.join(directory, "resumen_temas.txt")

            with open(file_path, "w", encoding="utf-8") as f:
                f.write("RESUMEN DE TEMAS\n")
                f.write("=" * 50 + "\n\n")

                for cluster in sorted(df["cluster"].unique()):
                    df_cluster = df[df["cluster"] == cluster]
                    f.write(f"\nTEMA {cluster+1}:\n")
                    f.write(f"- Mensajes: {len(df_cluster)}\n")
                    f.write(f"- Usuarios: {df_cluster['usuario'].nunique()}\n")
                    f.write(f"- Palabras clave: {', '.join(keywords[cluster][:10])}\n")
                    f.write("- Ejemplo mensaje:\n")
                    if not df_cluster.empty:
                        ejemplo = df_cluster.iloc[0]
                        f.write(
                            f"  {ejemplo['usuario']}: {ejemplo['mensaje_original'][:100]}...\n"
                        )
        except Exception as e:
            raise Exception(f"Error al guardar archivo de resumen: {str(e)}")
