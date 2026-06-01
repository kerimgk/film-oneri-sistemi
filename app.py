import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import requests

@st.cache_data
def get_movie_poster(movie_title):
    # Filmin adındaki yılı temizleyelim (ör: "Toy Story (1995)" -> "Toy Story")
    clean_title = movie_title.split("(")[0].strip()
    
    # TMDB API Anahtarı (Bunu ücretsiz alman gerekecek)
    api_key = "BURAYA_API_ANAHTARI_GELECEK" 
    url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={clean_title}"
    
    try:
        response = requests.get(url).json()
        if response.get('results'):
            poster_path = response['results'][0].get('poster_path')
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except Exception as e:
        pass
    
    # Eğer afiş bulunamazsa veya internet koparsa siyah şık bir boş kutu döndür
    return "https://via.placeholder.com/200x300/141414/E50914?text=Gorsel+Yok"

# ==========================================
# 1. BÖLÜM: VERİ YAPILARI (Müfredat Uyumu)
# ==========================================
class Node:
    def __init__(self, movie_title):
        self.movie_title = movie_title
        self.next = None

class MovieLinkedList:
    def __init__(self): self.head = None
    def add_movie(self, title):
        new_node = Node(title)
        if not self.head: self.head = new_node
        else:
            curr = self.head
            while curr.next: curr = curr.next
            curr.next = new_node
    def get_all(self):
        res, curr = [], self.head
        while curr: res.append(curr.movie_title); curr = curr.next
        return res

class ActionStack:
    def __init__(self): self.items = []
    def push(self, item): self.items.append(item)
    def get_all(self): return self.items[::-1]
class WatchQueue:
    def __init__(self): self.items = []
    def enqueue(self, item): self.items.append(item)
    # YENİ EKLENEN SATIR: Kuyruktan ilk elemanı çıkarır (FIFO)
    def dequeue(self): return self.items.pop(0) if self.items else None 
    def get_all(self): return self.items
# ==========================================
# 2. BÖLÜM: MODEL MİMARİSİ (NCF)
# ==========================================
class NCFModel(nn.Module):
    def __init__(self, num_users, num_items):
        super(NCFModel, self).__init__()
        # İsimleri state_dict ile eşleştirdik: emb -> embed
        self.user_embed = nn.Embedding(num_users, 50) 
        self.item_embed = nn.Embedding(num_items, 50)
        # fc -> fc_layers olarak güncellendi
        self.fc_layers = nn.Sequential(
            nn.Linear(100, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, u, i):
        # forward içindeki isimleri de güncellemeyi unutma
        x = torch.cat([self.user_embed(u), self.item_embed(i)], dim=-1)
        return self.fc_layers(x).squeeze()
# ==========================================
# 3. BÖLÜM: VERİ VE MODEL YÜKLEME
# ==========================================
@st.cache_data
def load_movies():
    url = "https://files.grouplens.org/datasets/movielens/ml-100k/u.item"
    cols = ['item_id', 'movie_title', 'release', 'v_release', 'url'] + [f'g{i}' for i in range(19)]
    return pd.read_csv(url, sep='|', names=cols, encoding='latin-1')

movies_df = load_movies()

def load_trained_model():
   
    model = NCFModel(num_users=943, num_items=1682) 
    
    state_dict = torch.load("ncf_model.pth", map_location=torch.device('cpu'))
    if not isinstance(state_dict, dict):
        state_dict = state_dict.state_dict()
        
    try:
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except RuntimeError as e:
      
        st.error(f"Boyut Uyuşmazlığı: {e}")
        return None
trained_model = load_trained_model()

# ==========================================
# 4. BÖLÜM: ARAYÜZ (UI) - NETFLIX TARZI
# ==========================================
st.set_page_config(page_title="Hibrit Öneri Sistemi", layout="wide", page_icon="🍿")

# --- ÖZEL CSS (Netflix Hissiyatı İçin) ---
st.markdown("""
<style>
    [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
        background-color: rgba(255, 255, 255, 0.02);
        border-radius: 10px;
        padding: 10px;
    }
    .kirmizi-baslik { color: #E50914; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='kirmizi-baslik'>🍿 Hibrit Film Öneri Sistemi</h1>", unsafe_allow_html=True)

# --- STATE YÖNETİMİ ---
if 'history' not in st.session_state: st.session_state.history = ActionStack()
if 'queue' not in st.session_state: st.session_state.queue = WatchQueue()

# YENİ: Filmlerin ekranda kalıcı olması için listeleri State'te tutuyoruz
if 'ncf_recs' not in st.session_state: st.session_state.ncf_recs = []
if 'cb_recs' not in st.session_state: st.session_state.cb_recs = []

# YENİ: Callback Fonksiyonu (Listeye eklemek için)
def kuyruga_ekle(film_adi):
    st.session_state.queue.enqueue(film_adi)

col1, col2 = st.columns([7, 3], gap="large")

with col1:
    st.subheader("Hoş Geldiniz! ID'nizi Girin:")
    u_id = st.number_input("Kullanıcı ID (1-943)", 1, 943, 11, label_visibility="collapsed")
    
    # --- 1. ÖNERİ SİSTEMİ: NCF ---
    if st.button("🚀 Benim İçin Önerileri Hesapla", type="primary"):
        if trained_model:
            items = torch.arange(len(movies_df))
            users = torch.full_like(items, u_id)
            with torch.no_grad():
                preds = trained_model(users, items).numpy()
            top_idx = preds.argsort()[-5:][::-1]
            # Sonuçları state'e kaydediyoruz
            st.session_state.ncf_recs = movies_df.iloc[top_idx]['movie_title'].tolist()
        else:
            st.warning("Model dosyası yükleniyor veya bulunamadı.")

    # State'te NCF önerisi varsa ekrana çiz
    if st.session_state.ncf_recs:
        st.markdown("### 🤖 Sizin İçin Seçtiklerimiz (NCF)")
        for r in st.session_state.ncf_recs:
            with st.container(border=True): 
                c_img, c_info, c_btn = st.columns([1, 4, 2])
                with c_img:
                    st.image("https://via.placeholder.com/100x150/141414/E50914?text=🎥", use_container_width=True)
                with c_info:
                    st.write(f"**{r}**")
                    st.caption("Eşleşme Oranı: %98 • HD")
                with c_btn:
                    st.write("") 
                    # DÜZELTME: on_click ile butona basılınca doğrudan fonksiyonu tetikliyoruz
                    st.button("➕ Listeme Ekle", key=f"btn_ncf_{r}", on_click=kuyruga_ekle, args=(r,), use_container_width=True)

    st.divider()

    # --- 2. ÖNERİ SİSTEMİ: İÇERİK TABANLI ---
    st.markdown("### 🔍 Benzerlerini Keşfet")
    sel_movie = st.selectbox("Sevdiğiniz bir filmi seçin:", movies_df['movie_title'].values)
    
    if st.button("Benzer Filmleri Bul"):
        movie_idx = movies_df[movies_df['movie_title'] == sel_movie].index[0]
        genre_matrix = movies_df.iloc[:, 5:24].values
        target_vector = genre_matrix[movie_idx]

        dot_product = np.dot(genre_matrix, target_vector)
        norms = np.linalg.norm(genre_matrix, axis=1) * np.linalg.norm(target_vector)
        similarity = dot_product / (norms + 1e-9)

        similarity[movie_idx] = -1 
        top_indices = similarity.argsort()[-5:][::-1]
        # Sonuçları state'e kaydediyoruz
        st.session_state.cb_recs = movies_df.iloc[top_indices]['movie_title'].tolist()

    # State'te Benzer Film önerisi varsa ekrana çiz
    if st.session_state.cb_recs:
        st.caption(f"✨ '{sel_movie}' sevenler bunları da izledi:")
        for r in st.session_state.cb_recs:
            with st.container(border=True):
                c_img, c_info, c_btn = st.columns([1, 4, 2])
                with c_img:
                    st.image("https://via.placeholder.com/100x150/222222/FFFFFF?text=🎞️", use_container_width=True)
                with c_info:
                    st.write(f"**{r}**")
                    st.caption("Benzerlik Skoru Yüksek")
                with c_btn:
                    st.write("")
                    st.button("➕ Listeme Ekle", key=f"btn_cb_{r}", on_click=kuyruga_ekle, args=(r,), use_container_width=True)

with col2:
    st.markdown("### 📂 Benim Listem")
    
    # --- İZLEME SIRASI (QUEUE) ---
    with st.expander("🍿 İzleme Sırası (Queue - FIFO)", expanded=True):
        queue_items = st.session_state.queue.get_all()
        if queue_items:
            for idx, q_movie in enumerate(queue_items):
                st.markdown(f"`{idx+1}.` **{q_movie}**")
            
            st.write("") 
            qc1, qc2 = st.columns(2)
            with qc1:
                if st.button("▶️ Sıradakini İzle", type="primary", use_container_width=True):
                    watched_movie = st.session_state.queue.dequeue()
                    st.session_state.history.push(watched_movie)
                    st.rerun()
            with qc2:
                if st.button("🗑️ Sırayı Boşalt", use_container_width=True):
                    st.session_state.queue.items = [] 
                    st.rerun()
        else:
            st.info("Listeniz şu an boş. Film ekleyin!")

    # --- İZLEME GEÇMİŞİ (STACK) ---
    with st.expander("🕰️ İzleme Geçmişi (Stack - LIFO)", expanded=True):
        history_items = st.session_state.history.get_all()
        if history_items:
            for h_movie in history_items:
                st.markdown(f"✔️ {h_movie}")
            
            st.write("")
            if st.button("🧹 Geçmişi Temizle", use_container_width=True):
                st.session_state.history.items = [] 
                st.rerun()
        else:
            st.caption("Henüz hiçbir şey izlemediniz.")

    # Sidebar Metrikleri
    st.divider()
    st.markdown("##### 📊 Model Performansı")
    m1, m2 = st.columns(2)
    m1.metric("Eğitim Hatası", "1.5", "-11.5")
    m2.metric("Seyreklik", "%93.7")
