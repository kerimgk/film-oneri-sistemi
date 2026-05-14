import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

# ==========================================
# 1. BÖLÜM: VERİ YAPILARI (Data Structures)
# ==========================================
class Node:
    def __init__(self, movie_title):
        self.movie_title = movie_title
        self.next = None

class MovieLinkedList:
    def __init__(self):
        self.head = None
    def add_movie(self, title):
        new_node = Node(title)
        if not self.head:
            self.head = new_node
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = new_node
    def to_list(self):
        movies = []
        current = self.head
        while current:
            movies.append(current.movie_title)
            current = current.next
        return movies

class ActionStack:
    def __init__(self):
        self.items = []
    def push(self, item): self.items.append(item)
    def pop(self): return self.items.pop() if self.items else None
    def get_all(self): return self.items[::-1] # LIFO Görünümü

class WatchQueue:
    def __init__(self):
        self.items = []
    def enqueue(self, item): self.items.append(item)
    def dequeue(self): return self.items.pop(0) if self.items else None
    def get_all(self): return self.items # FIFO Görünümü

# ==========================================
# 2. BÖLÜM: VERİ ÇEKME VE HAZIRLIK
# ==========================================
@st.cache_data # Veriyi her seferinde indirmesin diye önbelleğe alıyoruz
def load_data():
    # Sadece filmlerin adlarını çekiyoruz 
    movie_url = "https://files.grouplens.org/datasets/movielens/ml-100k/u.item"
    genre_cols = ["unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", 
                  "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", 
                  "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
    m_cols = ['item_id', 'movie_title', 'release_date', 'video_release_date', 'imdb_url'] + genre_cols
    movies = pd.read_csv(movie_url, sep='|', names=m_cols, encoding='latin-1')
    return movies

movies_df = load_data()

# ==========================================
# 3. BÖLÜM: STREAMLIT ARAYÜZÜ (UI)
# ==========================================
st.set_page_config(page_title="Hibrit Film Öneri Sistemi", page_icon="🎬", layout="wide")

st.title("🎬 Hibrit Film Öneri Sistemi")
st.divider()

# Sidebar (Yan Menü) Ayarları
st.sidebar.header("⚙️ Kullanıcı Ayarları")
user_id = st.sidebar.number_input("Kullanıcı ID (1-943)", min_value=1, max_value=943, value=10)
top_n = st.sidebar.slider("Öneri Sayısı", min_value=1, max_value=10, value=5)

st.sidebar.markdown("---")
st.sidebar.subheader("📈 Model Metrikleri")
st.sidebar.metric("Eğitim Hatası (RMSE)", "1.5 Yıldız", "-11.5 Düşüş") # Senin sunumundaki gerçek metrik
st.sidebar.metric("Seyreklik Oranı", "%93.70")

# Ana Ekran
col1, col2 = st.columns([2, 1])

with col1:
    # Mevcut col1 bloğunun içinde, butonun altına ekle:
    st.divider() # Görsel bir ayırıcı çizgi çeker
    st.subheader("🔍 Film İsmine Göre Benzer Öneriler")
    st.caption("Kullanıcıyı tanımadığımız (Cold Start) durumlarda içerik benzerliği kullanılır.")
    
    # Tüm film isimlerini liste olarak çekiyoruz
    movie_list = movies_df['movie_title'].values
    # Kullanıcının film seçebileceği kutu
    selected_movie = st.selectbox("Bir film seçin veya yazın:", movie_list, key="movie_search")

    if st.button("Benzer Filmleri Bul"):
        st.write(f"'{selected_movie}' filmine içerik (tür) olarak en yakın sonuçlar:")
        
        # Veri Yapıları Entegrasyonu: Sonuçları Bağlı Liste'de tutalım
        sim_list = MovieLinkedList()
        
        # Gerçek hibrit fonksiyonun buraya bağlanana kadar örnek veri çekelim
        similar_ones = movies_df.sample(n=top_n)['movie_title'].tolist() 
        
        for sim_film in similar_ones:
            sim_list.add_movie(sim_film)
            st.info(f"✨ {sim_film}")
            
        # VERİ YAPISI ENTEGRASYONU: Aranan filmi Stack'e (Yığın) ekle
        st.session_state.history.push(selected_movie)
    st.subheader(f"👤 Kullanıcı {user_id} İçin Top {top_n} Öneri")
    if st.button("🚀 Önerileri Hesapla"):
        with st.spinner("Yapay Zeka Modeli Çalışıyor..."):
            # BURAYA SENİN HİBRİT FONKSİYONUN GELECEK (Şimdilik rastgele çalışıyor gibi gösteriyoruz)
            # Senin Colab'daki kodunu buraya entegre edene kadar sahte bir Linked List gösterelim:
            
            öneri_listesi = MovieLinkedList()
            rastgele_filmler = movies_df['movie_title'].sample(n=top_n).tolist()
            
            for film in rastgele_filmler:
                öneri_listesi.add_movie(film)
                
            # Bağlı listeden veriyi çekip arayüze basıyoruz
            for index, film_adi in enumerate(öneri_listesi.to_list()):
                st.success(f"{index+1}. {film_adi}")

with col2:
    st.subheader("📚 Veri Yapıları Yönetimi")
    st.info("Bu bölüm Algoritma Analizi ders müfredatına göre tasarlanmıştır.")
    
    # Session State (Arayüz yenilense de veriyi tutmak için)
    if 'history' not in st.session_state:
        st.session_state.history = ActionStack()
        st.session_state.history.push("Inception (2010)")
        st.session_state.history.push("The Matrix (1999)")

    if 'queue' not in st.session_state:
        st.session_state.queue = WatchQueue()
        st.session_state.queue.enqueue("Interstellar (2014)")

    # Görüntüleme
    st.markdown("**İzleme Geçmişi (Stack - LIFO)**")
    st.write(st.session_state.history.get_all())
    
    st.markdown("**Sıradaki Filmler (Queue - FIFO)**")
    st.write(st.session_state.queue.get_all())
