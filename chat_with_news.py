import os
import re
import streamlit as st
from datetime import datetime, timedelta
from typing import Optional, Dict, Union, List

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.prompts import PromptTemplate
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
    FilterCondition,
)

# Ollama LLM Integration
from llama_index.llms.ollama import Ollama

# Local embeddings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Qdrant (local vector DB)
from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore


st.set_page_config(page_title="روبوت تحليل الأخبار", layout="centered")

# === Constants ===
DOC_FOLDER = "news_documents"
QDRANT_PATH = "qdrant_data"
QDRANT_COLLECTION = "news_multe5"
CHUNK_SIZE = 256
CHUNK_OVERLAP = 64
SIMILARITY_TOP_K = 8

# Ollama Configuration
OLLAMA_MODEL = "aya"  
OLLAMA_BASE_URL = "http://localhost:11434"  # Default Ollama URL

# === Prompt (concise, grounded, Arabic) ===
ARABIC_PROMPT_TMPL = (
    "أنت مساعد لتحليل الأخبار باللغة العربية الفصحى. تكلم فقط بالعربية، أجب فقط استنادًا إلى السياق."
    "- استخدم فقرات واضحة ومباشرة.\n"
    "- لا تضف معلومات خارجية.\n\n"
    "السياق:\n{context_str}\n\n"
    "السؤال:\n{query_str}\n"
    "الإجابة: "
)
PROMPT = PromptTemplate(ARABIC_PROMPT_TMPL)


def _extract_date_from_filename(filename: str):
    match = re.search(r"(\d{4}-\d{2}-\d{2})", filename)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y-%m-%d").date().isoformat()
    except Exception:
        return None


def _extract_date_from_query(query: str) -> Optional[Dict[str, Union[str, List[str]]]]:
    """
    Extract date information from query using rule-based patterns.
    Returns dict with type ('single', 'range') and dates.
    """
    query_lower = query.lower().strip()
    
    # Single date patterns
    single_patterns = [
        r'(\d{1,2})[-/](\d{1,2})[-/](\d{4})',
        r'(\d{4})[-/](\d{1,2})[-/](\d{1,2})',
        r'(\d{1,2})\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})',
        r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2}),?\s+(\d{4})',
    ]
    
    # Date range patterns
    range_patterns = [
        r'(\d{1,2})[-/](\d{1,2})[-/](\d{4})\s+(?:to|until|till|-)\s+(\d{1,2})[-/](\d{1,2})[-/](\d{4})',
        r'(\d{4})[-/](\d{1,2})[-/](\d{1,2})\s+(?:to|until|till|-)\s+(\d{4})[-/](\d{1,2})[-/](\d{1,2})',
        r'(last|this|next)\s+(week|month|year)',
        r'(yesterday|today|tomorrow)',
    ]
    
    # Try single date first
    for pattern in single_patterns:
        matches = re.findall(pattern, query_lower)
        if matches:
            for match in matches:
                try:
                    if len(match) == 3:
                        if len(match[0]) == 4:  # YYYY-MM-DD format
                            year, month, day = match
                        else:  # DD-MM-YYYY format
                            day, month, year = match
                        
                        # Handle month names
                        if isinstance(month, str) and month.isalpha():
                            month_map = {
                                'january': 1, 'february': 2, 'march': 3, 'april': 4,
                                'may': 5, 'june': 6, 'july': 7, 'august': 8,
                                'september': 9, 'october': 10, 'november': 11, 'december': 12
                            }
                            month = month_map.get(month.lower(), 1)
                        
                        date_obj = datetime(int(year), int(month), int(day))
                        return {
                            "type": "single",
                            "date": date_obj.date().isoformat(),
                            "query": query
                        }
                except (ValueError, TypeError):
                    continue
    
    # Try date ranges
    for pattern in range_patterns:
        matches = re.findall(pattern, query_lower)
        if matches:
            for match in matches:
                try:
                    today = datetime.now().date()
                    start_date = None
                    end_date = None
                    
                    if len(match) == 6 and all(x.isdigit() for x in match):
                        # Could be DD-MM-YYYY to DD-MM-YYYY or YYYY-MM-DD to YYYY-MM-DD
                        if int(match[0]) > 31:  # YYYY format
                            year1, month1, day1, year2, month2, day2 = match
                            start_date = datetime(int(year1), int(month1), int(day1))
                            end_date = datetime(int(year2), int(month2), int(day2))
                        else:  # DD format
                            day1, month1, year1, day2, month2, year2 = match
                            start_date = datetime(int(year1), int(month1), int(day1))
                            end_date = datetime(int(year2), int(month2), int(day2))
                    
                    elif len(match) == 2:  # Relative ranges
                        relative, period = match
                        
                        if period == "week":
                            if relative == "last":
                                end_date = today - timedelta(days=today.weekday() + 1)
                                start_date = end_date - timedelta(days=6)
                            elif relative == "this":
                                start_date = today - timedelta(days=today.weekday())
                                end_date = start_date + timedelta(days=6)
                            elif relative == "next":
                                start_date = today + timedelta(days=7-today.weekday())
                                end_date = start_date + timedelta(days=6)
                        elif period == "month":
                            if relative == "last":
                                start_date = (today.replace(day=1) - timedelta(days=1)).replace(day=1)
                                end_date = today.replace(day=1) - timedelta(days=1)
                            elif relative == "this":
                                start_date = today.replace(day=1)
                                end_date = (today.replace(day=1) + timedelta(days=32)).replace(day=1) - timedelta(days=1)
                            elif relative == "next":
                                start_date = (today.replace(day=1) + timedelta(days=32)).replace(day=1)
                                end_date = (start_date + timedelta(days=32)).replace(day=1) - timedelta(days=1)
                        elif period == "year":
                            if relative == "last":
                                start_date = today.replace(month=1, day=1, year=today.year-1)
                                end_date = today.replace(month=12, day=31, year=today.year-1)
                            elif relative == "this":
                                start_date = today.replace(month=1, day=1)
                                end_date = today.replace(month=12, day=31)
                            elif relative == "next":
                                start_date = today.replace(month=1, day=1, year=today.year+1)
                                end_date = today.replace(month=12, day=31, year=today.year+1)
                    
                    elif len(match) == 1:  # Yesterday, today, tomorrow
                        relative = match[0]
                        
                        if relative == "yesterday":
                            start_date = end_date = today - timedelta(days=1)
                        elif relative == "today":
                            start_date = end_date = today
                        elif relative == "tomorrow":
                            start_date = end_date = today + timedelta(days=1)
                    
                    if start_date and end_date:
                        return {
                            "type": "range",
                            "start": start_date.date().isoformat() if isinstance(start_date, datetime) else start_date.isoformat(),
                            "end": end_date.date().isoformat() if isinstance(end_date, datetime) else end_date.isoformat(),
                            "query": query
                        }
                except (ValueError, TypeError) as e:
                    continue
    
    return None


def _load_documents_with_metadata():
    """Load documents and add date metadata from filenames."""
    if not os.path.exists(DOC_FOLDER):
        st.error(f"⚠️ مجلد المستندات '{DOC_FOLDER}' غير موجود.")
        st.stop()
    
    all_files = [
        f for f in os.listdir(DOC_FOLDER)
        if f.endswith(".docx") and not f.startswith("~$")
    ]
    
    if not all_files:
        st.error("⚠️ لا توجد ملفات .docx صالحة في مجلد المستندات.")
        st.stop()

    docs = SimpleDirectoryReader(
        input_dir=DOC_FOLDER,
        recursive=True,
    ).load_data()

    # Add date metadata from filename
    for d in docs:
        path = d.metadata.get("file_path") or d.metadata.get("filename") or ""
        fname = os.path.basename(path)
        date_iso = _extract_date_from_filename(fname)
        if date_iso:
            d.metadata["date"] = date_iso
    
    return docs


def _init_models():
    """Initialize Ollama LLM and local embeddings."""
    try:
        # Initialize Ollama LLM
        llm = Ollama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            request_timeout=120.0,
            temperature=0.7,
        )
        
        # Test Ollama connection
        try:
            test_response = llm.complete("test")
            st.sidebar.success(f"✅ متصل بـ Ollama: {OLLAMA_MODEL}")
        except Exception as e:
            st.error(f"⚠️ فشل الاتصال بـ Ollama. تأكد من تشغيل Ollama وتنزيل النموذج '{OLLAMA_MODEL}'")
            st.error(f"الخطأ: {str(e)}")
            st.stop()
        
        # Initialize local embeddings
        embed = HuggingFaceEmbedding(
            model_name="akhooli/Arabic-SBERT-100K",
            device="cpu"    #if st.session_state.get("use_gpu", True) else "cpu"
        )
        
        # Configure LlamaIndex Settings
        Settings.llm = llm
        Settings.embed_model = embed
        Settings.chunk_size = CHUNK_SIZE
        Settings.chunk_overlap = CHUNK_OVERLAP
        
        return llm, embed
        
    except Exception as e:
        st.error(f"⚠️ خطأ في تهيئة النماذج: {str(e)}")
        st.stop()


def _init_qdrant_vector_store():
    """Initialize Qdrant vector store."""
    client = QdrantClient(path=QDRANT_PATH)
    vector_store = QdrantVectorStore(client=client, collection_name=QDRANT_COLLECTION)
    return client, vector_store


@st.cache_resource(show_spinner=False)
def load_or_build_index():
    """Load existing index or build new one."""
    llm, embed = _init_models()
    client, vector_store = _init_qdrant_vector_store()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Check if collection exists
    collection_exists = False
    try:
        client.get_collection(QDRANT_COLLECTION)
        collection_exists = True
    except Exception:
        collection_exists = False

    if collection_exists:
        # Load from existing vector store
        index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed)
        st.sidebar.info(f"📚 تم تحميل الفهرس الموجود: {QDRANT_COLLECTION}")
        return index

    # Build new index
    with st.spinner("🔨 بناء الفهرس للمرة الأولى..."):
        docs = _load_documents_with_metadata()
        node_parser = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        nodes = node_parser.get_nodes_from_documents(docs)
        index = VectorStoreIndex(nodes=nodes, storage_context=storage_context, embed_model=embed)
        st.sidebar.success(f"✅ تم بناء الفهرس بنجاح: {len(nodes)} عقدة")
        return index


def _build_query_engine(index: VectorStoreIndex, date_filter: Optional[Dict] = None):
    """Build query engine with optional date filtering."""
    
    if date_filter:
        # Build retrieval-time metadata filters so only nodes with matching dates are fetched
        if date_filter.get("type") == "single":
            filters = MetadataFilters(
                filters=[
                    MetadataFilter(key="date", operator=FilterOperator.EQ, value=date_filter["date"]),
                ],
                condition=FilterCondition.AND,
            )
        else:
            # Range: start <= date <= end
            filters = MetadataFilters(
                filters=[
                    MetadataFilter(key="date", operator=FilterOperator.GTE, value=date_filter["start"]),
                    MetadataFilter(key="date", operator=FilterOperator.LTE, value=date_filter["end"]),
                ],
                condition=FilterCondition.AND,
            )

        # Use retrieval-time filters directly in the query engine
        query_engine = index.as_query_engine(
            text_qa_template=PROMPT,
            response_mode="compact",
            similarity_top_k=SIMILARITY_TOP_K,
            filters=filters,
            verbose=False,
        )
        
        # Wrap query method to filter by date (defensive fallback if any node lacks 'date')
        original_query = query_engine.query
        
        def filtered_query(query_str: str):
            response = original_query(query_str)
            
            # Filter source nodes by date
            if hasattr(response, 'source_nodes') and response.source_nodes:
                filtered_nodes = []
                for node in response.source_nodes:
                    node_date = node.metadata.get("date")
                    # Fallback: derive date from filename if missing on the node
                    if not node_date:
                        path = node.metadata.get("file_path") or node.metadata.get("filename")
                        if path:
                            node_date = _extract_date_from_filename(os.path.basename(path))
                    if node_date:
                        if date_filter["type"] == "single":
                            if node_date == date_filter["date"]:
                                filtered_nodes.append(node)
                        elif date_filter["type"] == "range":
                            if date_filter["start"] <= node_date <= date_filter["end"]:
                                filtered_nodes.append(node)
                
                filtered_nodes = filtered_nodes[:SIMILARITY_TOP_K]
                response.source_nodes = filtered_nodes
                
                # If we have matches, regenerate the answer using ONLY filtered context
                if filtered_nodes:
                    try:
                        context_str = "\n\n".join([n.get_text() for n in filtered_nodes])
                        prompt_str = PROMPT.format(context_str=context_str, query_str=query_str)
                        llm = Settings.llm
                        regenerated = llm.complete(prompt_str)
                        # Some LLM wrappers return objects; get text accordingly
                        regenerated_text = getattr(regenerated, "text", str(regenerated))
                        response.response = regenerated_text
                    except Exception:
                        # If regeneration fails, keep the original response text
                        pass
                else:
                    # Update response if no matches
                    if date_filter["type"] == "single":
                        response.response = f"لا توجد أخبار متاحة للتاريخ: {date_filter['date']}"
                    else:
                        response.response = f"لا توجد أخبار متاحة للفترة من {date_filter['start']} إلى {date_filter['end']}"
            
            return response
        
        query_engine.query = filtered_query
        return query_engine

    # Standard query engine without date filtering
    return index.as_query_engine(
        text_qa_template=PROMPT,
        response_mode="compact",
        similarity_top_k=SIMILARITY_TOP_K,
        verbose=False,
    )


# === UI ===
st.title("📰 روبوت تحليل الأخبارـ ")
st.markdown("**اكتب سؤالك أدناه. تعتمد الإجابات فقط على الوثائق المخزنة محليًا.**")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ الإعدادات")
    st.info(f"**النموذج:** {OLLAMA_MODEL}")
    st.info(f"**قاعدة البيانات:** {QDRANT_COLLECTION}")
    
    if st.button("🔄 إعادة بناء الفهرس"):
        st.cache_resource.clear()
        st.rerun()

# Initialize index
index = load_or_build_index()

# Session state for conversation history
if "last_response" not in st.session_state:
    st.session_state.last_response = ""
if "last_query" not in st.session_state:
    st.session_state.last_query = ""

# Show last Q&A
if st.session_state.last_response:
    st.markdown("#### ❓ سؤالك السابق:")
    st.markdown(f"`{st.session_state.last_query}`")
    st.markdown("#### 🤖 الرد السابق:")
    st.markdown(st.session_state.last_response)
    st.markdown("---")

# Query input
query = st.text_input("🧑‍💻 أنت:", placeholder="اكتب سؤالك هنا...", key="user_query")

if query:
    # Extract date information
    date_info = _extract_date_from_query(query)
    
    # Show date filter info
    if date_info:
        if date_info["type"] == "single":
            st.info(f"📅 تم اكتشاف تاريخ: {date_info['date']}")
        else:
            st.info(f"📅 تم اكتشاف نطاق تاريخ: من {date_info['start']} إلى {date_info['end']}")
    
    # Build query engine
    query_engine = _build_query_engine(index, date_info)
    
    with st.spinner("🔍 جاري التحليل  ..."):
        try:
            response = query_engine.query(query)
            answer_text = getattr(response, "response", str(response))
            
            # Update session state
            st.session_state.last_response = answer_text
            st.session_state.last_query = query

            # Display response
            st.markdown("### 🤖 الرد:")
            st.markdown(answer_text)

            # Show sources
            source_files = []
            if hasattr(response, "source_nodes") and response.source_nodes:
                for node in response.source_nodes:
                    path = node.metadata.get("file_path") or node.metadata.get("filename")
                    if path:
                        source_files.append(os.path.basename(path))
            
            if source_files:
                unique_sorted = ", ".join(sorted(set(source_files)))
                st.markdown(f"📂 **المصادر المستخدمة:** {unique_sorted}")
            
            # Display metadata in expander
            with st.expander("📊 تفاصيل إضافية"):
                st.write(f"**عدد المصادر:** {len(source_files)}")
                if date_info:
                    st.write(f"**فلترة التاريخ:** نعم")
                else:
                    st.write(f"**فلترة التاريخ:** لا")

        except Exception as e:
            st.error(f"⚠️ حدث خطأ أثناء المعالجة: {e}")
            st.info("💡 تأكد من أن Ollama يعمل بشكل صحيح وأن النموذج متاح.")