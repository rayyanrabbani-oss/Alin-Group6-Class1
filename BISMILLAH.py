import streamlit as st
import numpy as np
import cv2
from PIL import Image
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import io
import warnings
warnings.filterwarnings('ignore')

# ==================================================
#  FUNGSI BANTU
# ==================================================
def load_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    return np.array(img)

def apply_transformation_full(image, matrix):
    h, w = image.shape[:2]
    corners = np.array([[0,0],[w,0],[0,h],[w,h]], dtype=np.float32)
    new_corners = cv2.transform(np.array([corners]), matrix)[0]
    x_coords, y_coords = new_corners[:,0], new_corners[:,1]
    min_x, max_x = x_coords.min(), x_coords.max()
    min_y, max_y = y_coords.min(), y_coords.max()
    new_w, new_h = int(max_x - min_x), int(max_y - min_y)
    translation = np.array([[1,0,-min_x],[0,1,-min_y]], dtype=np.float32)
    full_matrix = translation @ np.vstack([matrix,[0,0,1]])
    return cv2.warpAffine(image, full_matrix[:2,:], (new_w,new_h))

def apply_translation(image, tx, ty):
    h, w = image.shape[:2]
    margin = max(abs(tx), abs(ty), 300)
    new_w, new_h = w + margin*2, h + margin*2
    matrix = np.float32([[1,0,tx+margin],[0,1,ty+margin]])
    return cv2.warpAffine(image, matrix, (new_w,new_h))

# ==================================================
#  FUNGSI KONVOLUSI KUSTOM
# ==================================================
def apply_custom_convolution(image, kernel):
    """Menerapkan kernel konvolusi kustom pada gambar"""
    img_float = image.astype(np.float32) / 255.0
    
    if len(image.shape) == 3:
        result = np.zeros_like(img_float)
        for i in range(3):
            result[:,:,i] = cv2.filter2D(img_float[:,:,i], -1, kernel)
    else:
        result = cv2.filter2D(img_float, -1, kernel)
    
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    return result

def create_blur_kernel(kernel_size=3):
    """Membuat kernel blur (smoothing filter)"""
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
    kernel /= kernel_size * kernel_size
    return kernel

def create_sharpen_kernel(strength=1.0):
    """Membuat kernel sharpen (high-pass filter)"""
    kernel = np.array([[0, -strength, 0],
                       [-strength, 4*strength+1, -strength],
                       [0, -strength, 0]], dtype=np.float32)
    return kernel

def create_edge_detection_kernel():
    """Membuat kernel edge detection"""
    kernel = np.array([[-1, -1, -1],
                       [-1, 8, -1],
                       [-1, -1, -1]], dtype=np.float32)
    return kernel

def create_emboss_kernel():
    """Membuat kernel emboss"""
    kernel = np.array([[-2, -1, 0],
                       [-1, 1, 1],
                       [0, 1, 2]], dtype=np.float32)
    return kernel

# ==================================================
#  FUNGSI BACKGROUND REMOVAL MULTI-METHOD
# ==================================================

# METODE 1: GRABCUT (Sangat Akurat)
def remove_background_grabcut(image, rect=None, iterations=5):
    """Menggunakan GrabCut algorithm untuk segmentasi"""
    mask = np.zeros(image.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    if rect is None:
        h, w = image.shape[:2]
        rect = (int(w*0.1), int(h*0.1), int(w*0.8), int(h*0.8))
    
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    result = image * mask2[:, :, np.newaxis]
    b, g, r = cv2.split(result)
    rgba = cv2.merge([b, g, r, mask2 * 255])
    
    return result, rgba, mask2 * 255

# METODE 2: CANNY EDGE + FLOOD FILL
def remove_background_canny_floodfill(image):
    """Menggunakan edge detection dan flood fill"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    mask = np.zeros_like(gray)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest_contour], -1, 255, -1)
    
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    result = cv2.bitwise_and(image, image, mask=mask)
    b, g, r = cv2.split(result)
    rgba = cv2.merge([b, g, r, mask])
    
    return result, rgba, mask

# METODE 3: IMPROVED HSV METHOD
def remove_background_improved_hsv(image):
    """Metode HSV yang lebih canggih"""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, w = image.shape[:2]
    
    # Sample edges
    edge_samples = []
    edge_samples.extend(image[0, :])
    edge_samples.extend(image[-1, :])
    edge_samples.extend(image[:, 0])
    edge_samples.extend(image[:, -1])
    
    edge_samples = np.array(edge_samples)
    avg_color = np.median(edge_samples, axis=0).astype(np.uint8)
    avg_color_hsv = cv2.cvtColor(np.uint8([[avg_color]]), cv2.COLOR_RGB2HSV)[0][0]
    
    # Calculate dynamic thresholds
    h_mean, h_std = np.mean(hsv[:,:,0]), np.std(hsv[:,:,0])
    s_mean, s_std = np.mean(hsv[:,:,1]), np.std(hsv[:,:,1])
    v_mean, v_std = np.mean(hsv[:,:,2]), np.std(hsv[:,:,2])
    
    lower_hsv = np.array([
        max(0, avg_color_hsv[0] - int(h_std * 1.5)),
        max(0, avg_color_hsv[1] - int(s_std * 2)),
        max(0, avg_color_hsv[2] - int(v_std * 2))
    ])
    
    upper_hsv = np.array([
        min(179, avg_color_hsv[0] + int(h_std * 1.5)),
        min(255, avg_color_hsv[1] + int(s_std * 2)),
        min(255, avg_color_hsv[2] + int(v_std * 2))
    ])
    
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    mask = cv2.bitwise_not(mask)
    
    # Find largest connected component
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        sizes = stats[1:, -1]
        largest_label = np.argmax(sizes) + 1
        mask = np.zeros_like(mask)
        mask[labels == largest_label] = 255
    
    result = cv2.bitwise_and(image, image, mask=mask)
    b, g, r = cv2.split(result)
    rgba = cv2.merge([b, g, r, mask])
    
    return result, rgba, mask, (lower_hsv, upper_hsv)

# METODE 4: HYBRID K-MEANS
def remove_background_hybrid(image, n_clusters=3):
    """Metode hybrid K-Means"""
    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels)
    clustered = labels.reshape(image.shape[:2])
    
    from scipy import ndimage
    labeled_array, num_features = ndimage.label(clustered)
    sizes = ndimage.sum(clustered, labeled_array, range(num_features + 1))
    
    if len(sizes) > 1:
        background_label = sizes[1:].argmax() + 1
        mask = (labeled_array != background_label).astype(np.uint8) * 255
    else:
        mask = np.ones_like(clustered, dtype=np.uint8) * 255
    
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    result = cv2.bitwise_and(image, image, mask=mask)
    b, g, r = cv2.split(result)
    rgba = cv2.merge([b, g, r, mask])
    
    return result, rgba, mask

def create_checkerboard_background(h, w, checker_size=20):
    """Membuat background checkerboard"""
    background = np.ones((h, w, 3), dtype=np.uint8) * 255
    for i in range(0, h, checker_size*2):
        for j in range(0, w, checker_size*2):
            background[i:i+checker_size, j:j+checker_size] = [200, 200, 200]
            background[i+checker_size:i+checker_size*2, j+checker_size:j+checker_size*2] = [200, 200, 200]
    return background

# ==================================================
#  CUSTOM CSS
# ==================================================
st.markdown("""
<style>
.main-header {
    text-align: center;
    color: #2c3e50;
    margin-bottom: 30px;
}
.method-card {
    background: white;
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    border-left: 5px solid #4f8bf9;
}
.tips-box {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 20px;
    border-radius: 15px;
    color: white;
    margin: 20px 0;
}
.result-card {
    background: #f8f9fa;
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
    border: 1px solid #dee2e6;
}
.tool-card {
    background: white;
    border-radius: 10px;
    padding: 20px;
    margin: 15px 0;
    box-shadow: 0 3px 15px rgba(0,0,0,0.1);
    transition: transform 0.3s;
}
.tool-card:hover {
    transform: translateY(-5px);
}
</style>
""", unsafe_allow_html=True)

# ==================================================
#  GLOBAL LANGUAGE SELECTION
# ==================================================
language = st.sidebar.selectbox("Choose Language / Pilih Bahasa", ["Indonesia", "English"])

# ==================================================
#  PAGE 1 — HOME
# ==================================================
def page_home():
    if language == "Indonesia":
        st.title("📸 Aplikasi Transformasi & Pengolahan Gambar")
        st.markdown("""
        ### **Selamat Datang di Aplikasi Pengolahan Gambar!**
        
        Aplikasi ini menyediakan berbagai alat untuk:
        
        **🛠️ Image Tools:**
        - Translasi, Scaling, Rotasi
        - Shearing, Refleksi
        - Filter Konvolusi (Blur, Sharpen, Edge Detection)
        
        **🎭 Advanced Background Removal:**
        - 4 Metode berbeda untuk hasil optimal
        - GrabCut Algorithm (paling akurat)
        - Canny Edge + Flood Fill
        - Improved HSV Method
        - Hybrid K-Means Clustering
        
        **💡 Tips:**
        - Untuk background removal, coba **GrabCut** terlebih dahulu
        - Gunakan **Image Tools** untuk transformasi dasar
        - Semua hasil bisa didownload dalam format PNG
        """)
    else:
        st.title("📸 Image Transformation & Processing App")
        st.markdown("""
        ### **Welcome to Image Processing Application!**
        
        This app provides various tools for:
        
        **🛠️ Image Tools:**
        - Translation, Scaling, Rotation
        - Shearing, Reflection
        - Convolution Filters (Blur, Sharpen, Edge Detection)
        
        **🎭 Advanced Background Removal:**
        - 4 Different methods for optimal results
        - GrabCut Algorithm (most accurate)
        - Canny Edge + Flood Fill
        - Improved HSV Method
        - Hybrid K-Means Clustering
        
        **💡 Tips:**
        - For background removal, try **GrabCut** first
        - Use **Image Tools** for basic transformations
        - All results can be downloaded in PNG format
        """)

# ==================================================
#  PAGE 2 — IMAGE TOOLS
# ==================================================
def page_image_tools():
    if language == "Indonesia":
        st.title("🛠️ Alat Pengolahan Gambar")
        menu_label = "Pilih Tool:"
        upload_label = "Unggah gambar"
        info_label = "Unggah gambar untuk mulai menggunakan tools."
        before_after_label = "Perbandingan Hasil"
        menu_options = ["Translasi", "Scaling", "Rotasi", "Shearing", "Refleksi", 
                       "Konvolusi (Filter)"]
    else:
        st.title("🛠️ Image Processing Tools")
        menu_label = "Select Tool:"
        upload_label = "Upload image"
        info_label = "Upload an image to start using the tools."
        before_after_label = "Result Comparison"
        menu_options = ["Translation", "Scaling", "Rotation", "Shearing", "Reflection",
                       "Convolution (Filters)"]

    menu = st.sidebar.radio(menu_label, menu_options)
    uploaded = st.file_uploader(upload_label, type=["jpg", "png", "jpeg", "bmp"])

    if not uploaded:
        st.info(info_label)
        return

    img = load_image(uploaded)
    
    st.markdown(f"### {before_after_label}")
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(img, caption="🖼️ " + ("Gambar Asli" if language=="Indonesia" else "Original Image"), 
                use_container_width=True)

    # === TRANSLASI ===
    if menu in ["Translasi", "Translation"]:
        if language == "Indonesia":
            st.markdown("""
            <div class="tool-card">
            <h3>📤 Translasi (Pergeseran)</h3>
            <p>Geser gambar secara horizontal (X) dan vertikal (Y)</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="tool-card">
            <h3>📤 Translation (Shifting)</h3>
            <p>Shift image horizontally (X) and vertically (Y)</p>
            </div>
            """, unsafe_allow_html=True)
        
        col_x, col_y = st.columns(2)
        with col_x:
            tx = st.slider("Pergeseran X" if language=="Indonesia" else "Shift X", 
                          -500, 500, 0, 10)
        with col_y:
            ty = st.slider("Pergeseran Y" if language=="Indonesia" else "Shift Y", 
                          -500, 500, 0, 10)
        
        result = apply_translation(img, tx, ty)
        
        with col2:
            st.image(result, caption="↕️ " + ("Hasil Translasi" if language=="Indonesia" else "Translation Result"), 
                    use_container_width=True)

    # === SCALING ===
    elif menu in ["Scaling", "Scaling"]:
        if language == "Indonesia":
            st.markdown("""
            <div class="tool-card">
            <h3>📏 Scaling (Perubahan Skala)</h3>
            <p>Ubah ukuran gambar dengan faktor skala</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="tool-card">
            <h3>📏 Scaling (Size Change)</h3>
            <p>Change image size with scale factor</p>
            </div>
            """, unsafe_allow_html=True)
        
        col_sx, col_sy = st.columns(2)
        with col_sx:
            sx = st.slider("Skala X" if language=="Indonesia" else "Scale X", 
                          0.1, 3.0, 1.0, 0.1)
        with col_sy:
            sy = st.slider("Skala Y" if language=="Indonesia" else "Scale Y", 
                          0.1, 3.0, 1.0, 0.1)
        
        matrix = np.float32([[sx, 0, 0], [0, sy, 0]])
        result = apply_transformation_full(img, matrix)
        
        with col2:
            st.image(result, caption="⚖️ " + ("Hasil Scaling" if language=="Indonesia" else "Scaling Result"), 
                    use_container_width=True)

    # === ROTASI ===
    elif menu in ["Rotasi", "Rotation"]:
        if language == "Indonesia":
            st.markdown("""
            <div class="tool-card">
            <h3>🔄 Rotasi</h3>
            <p>Putar gambar dengan sudut tertentu</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="tool-card">
            <h3>🔄 Rotation</h3>
            <p>Rotate image with specific angle</p>
            </div>
            """, unsafe_allow_html=True)
        
        angle = st.slider("Sudut Rotasi (derajat)" if language=="Indonesia" else "Rotation Angle (degrees)", 
                         0, 360, 0, 1)
        rad = np.radians(angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        matrix = np.float32([[cos_a, -sin_a, 0], [sin_a, cos_a, 0]])
        result = apply_transformation_full(img, matrix)
        
        with col2:
            st.image(result, caption="🌀 " + ("Hasil Rotasi" if language=="Indonesia" else "Rotation Result"), 
                    use_container_width=True)

    # === SHEARING ===
    elif menu in ["Shearing", "Shearing"]:
        if language == "Indonesia":
            st.markdown("""
            <div class="tool-card">
            <h3>✂️ Shearing (Geser)</h3>
            <p>Geser gambar secara diagonal</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="tool-card">
            <h3>✂️ Shearing (Skew)</h3>
            <p>Skew image diagonally</p>
            </div>
            """, unsafe_allow_html=True)
        
        col_shx, col_shy = st.columns(2)
        with col_shx:
            shx = st.slider("Shear X" if language=="Indonesia" else "Shear X", 
                           -1.0, 1.0, 0.0, 0.1)
        with col_shy:
            shy = st.slider("Shear Y" if language=="Indonesia" else "Shear Y", 
                           -1.0, 1.0, 0.0, 0.1)
        
        matrix = np.float32([[1, shx, 0], [shy, 1, 0]])
        result = apply_transformation_full(img, matrix)
        
        with col2:
            st.image(result, caption="📐 " + ("Hasil Shearing" if language=="Indonesia" else "Shearing Result"), 
                    use_container_width=True)

    # === REFLEKSI ===
    elif menu in ["Refleksi", "Reflection"]:
        if language == "Indonesia":
            st.markdown("""
            <div class="tool-card">
            <h3>🪞 Refleksi (Pencerminan)</h3>
            <p>Balik gambar secara horizontal atau vertikal</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="tool-card">
            <h3>🪞 Reflection (Mirror)</h3>
            <p>Flip image horizontally or vertically</p>
            </div>
            """, unsafe_allow_html=True)
        
        reflection_type = st.radio(
            "Jenis Refleksi:" if language=="Indonesia" else "Reflection Type:",
            ["Horizontal", "Vertical", "Kedua-duanya" if language=="Indonesia" else "Both"]
        )
        
        if reflection_type == "Horizontal":
            matrix = np.float32([[-1, 0, 0], [0, 1, 0]])
        elif reflection_type == "Vertical":
            matrix = np.float32([[1, 0, 0], [0, -1, 0]])
        else:  # Both
            matrix = np.float32([[-1, 0, 0], [0, -1, 0]])
        
        result = apply_transformation_full(img, matrix)
        
        with col2:
            st.image(result, caption="🪞 " + ("Hasil Refleksi" if language=="Indonesia" else "Reflection Result"), 
                    use_container_width=True)

    # === KONVOLUSI (FILTER) ===
    elif menu in ["Konvolusi (Filter)", "Convolution (Filters)"]:
        if language == "Indonesia":
            st.markdown("""
            <div class="tool-card">
            <h3>🎛️ Filter Konvolusi</h3>
            <p>Terapkan berbagai filter pada gambar</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="tool-card">
            <h3>🎛️ Convolution Filters</h3>
            <p>Apply various filters to image</p>
            </div>
            """, unsafe_allow_html=True)
        
        filter_type = st.selectbox(
            "Pilih Filter:" if language=="Indonesia" else "Select Filter:",
            ["Blur", "Sharpen", "Edge Detection", "Emboss", "Kustom"]
        )
        
        if filter_type == "Blur":
            kernel_size = st.slider("Ukuran Kernel Blur" if language=="Indonesia" else "Blur Kernel Size", 
                                    3, 15, 5, 2)
            kernel = create_blur_kernel(kernel_size)
            
        elif filter_type == "Sharpen":
            strength = st.slider("Kekuatan Sharpen" if language=="Indonesia" else "Sharpen Strength", 
                                0.1, 3.0, 1.0, 0.1)
            kernel = create_sharpen_kernel(strength)
            
        elif filter_type == "Edge Detection":
            kernel = create_edge_detection_kernel()
            
        elif filter_type == "Emboss":
            kernel = create_emboss_kernel()
            
        else:  # Kustom
            if language == "Indonesia":
                st.write("Masukkan kernel kustom (3x3):")
            else:
                st.write("Enter custom kernel (3x3):")
            
            col1_k, col2_k, col3_k = st.columns(3)
            with col1_k:
                k11 = st.number_input("k11", value=0.0)
                k21 = st.number_input("k21", value=0.0)
                k31 = st.number_input("k31", value=0.0)
            with col2_k:
                k12 = st.number_input("k12", value=0.0)
                k22 = st.number_input("k22", value=1.0)
                k32 = st.number_input("k32", value=0.0)
            with col3_k:
                k13 = st.number_input("k13", value=0.0)
                k23 = st.number_input("k23", value=0.0)
                k33 = st.number_input("k33", value=0.0)
            
            kernel = np.array([[k11, k12, k13],
                              [k21, k22, k23],
                              [k31, k32, k33]], dtype=np.float32)
        
        result = apply_custom_convolution(img, kernel)
        
        with col2:
            caption_map = {
                "Blur": "🟡 " + ("Hasil Blur" if language=="Indonesia" else "Blur Result"),
                "Sharpen": "🔍 " + ("Hasil Sharpen" if language=="Indonesia" else "Sharpen Result"),
                "Edge Detection": "📐 " + ("Deteksi Tepi" if language=="Indonesia" else "Edge Detection"),
                "Emboss": "🏛️ " + ("Hasil Emboss" if language=="Indonesia" else "Emboss Result"),
                "Kustom": "🎨 " + ("Filter Kustom" if language=="Indonesia" else "Custom Filter")
            }
            st.image(result, caption=caption_map[filter_type], use_container_width=True)
    
    # Download button untuk semua tools
    st.markdown("---")
    if language == "Indonesia":
        st.subheader("📥 Download Hasil")
    else:
        st.subheader("📥 Download Result")
    
    result_bytes = cv2.imencode('.png', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))[1].tobytes()
    st.download_button(
        label="⬇️ Download PNG" if language=="Indonesia" else "⬇️ Download PNG",
        data=result_bytes,
        file_name=f"processed_{menu.lower().replace(' ', '_')}.png",
        mime="image/png",
        use_container_width=True
    )

# ==================================================
#  PAGE 3 — BACKGROUND REMOVAL
# ==================================================
def page_background_removal():
    if language == "Indonesia":
        st.title("🎭 Advanced Background Removal")
        st.markdown("""
        <div class="tips-box">
        <h3>🚀 Panduan Cepat:</h3>
        <ul>
        <li><b>Coba GrabCut terlebih dahulu</b> - paling akurat</li>
        <li><b>Atur rectangle</b> untuk menentukan area objek</li>
        <li><b>Coba semua metode</b> jika satu belum optimal</li>
        <li><b>Download hasil transparan</b> untuk editing lebih lanjut</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.title("🎭 Advanced Background Removal")
        st.markdown("""
        <div class="tips-box">
        <h3>🚀 Quick Guide:</h3>
        <ul>
        <li><b>Try GrabCut first</b> - most accurate</li>
        <li><b>Adjust rectangle</b> to define object area</li>
        <li><b>Try all methods</b> if one is not optimal</li>
        <li><b>Download transparent result</b> for further editing</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Upload gambar
    uploaded = st.file_uploader(
        "📤 Upload Gambar" if language == "Indonesia" else "📤 Upload Image", 
        type=["jpg", "png", "jpeg", "bmp"]
    )
    
    if not uploaded:
        st.info("ℹ️ " + ("Silakan upload gambar untuk memulai" if language == "Indonesia" else "Please upload an image to start"))
        return
    
    # Load image
    img = load_image(uploaded)
    
    # Tampilkan gambar asli
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="🖼️ " + ("Gambar Asli" if language == "Indonesia" else "Original Image"), use_container_width=True)
    
    # Pilih metode
    st.markdown("---")
    if language == "Indonesia":
        st.subheader("🛠️ Pilih Metode Background Removal")
        method_options = [
            "🎯 GrabCut Algorithm (Rekomendasi)",
            "🌊 Canny Edge + Flood Fill", 
            "🌈 Improved HSV Method",
            "🧩 Hybrid K-Means Clustering"
        ]
    else:
        st.subheader("🛠️ Select Background Removal Method")
        method_options = [
            "🎯 GrabCut Algorithm (Recommended)",
            "🌊 Canny Edge + Flood Fill", 
            "🌈 Improved HSV Method",
            "🧩 Hybrid K-Means Clustering"
        ]
    
    selected_method = st.selectbox(
        "Pilih metode:" if language == "Indonesia" else "Select method:",
        method_options
    )
    
    # Parameters berdasarkan metode
    if "GrabCut" in selected_method:
        if language == "Indonesia":
            st.markdown("""
            <div class="method-card">
            <h4>🎯 GrabCut Algorithm</h4>
            <p><b>Cocok untuk:</b> Semua jenis gambar, terutama yang kompleks</p>
            <p><b>Atur rectangle</b> untuk menentukan area objek</p>
            </div>
            """, unsafe_allow_html=True)
            
            col_a, col_b = st.columns(2)
            with col_a:
                rect_x = st.slider("Posisi X", 0, img.shape[1], int(img.shape[1]*0.1))
                rect_y = st.slider("Posisi Y", 0, img.shape[0], int(img.shape[0]*0.1))
            with col_b:
                rect_w = st.slider("Lebar", 10, img.shape[1], int(img.shape[1]*0.8))
                rect_h = st.slider("Tinggi", 10, img.shape[0], int(img.shape[0]*0.8))
                iterations = st.slider("Iterasi", 1, 10, 5)
            
            rect = (rect_x, rect_y, rect_w, rect_h)
            
        else:
            st.markdown("""
            <div class="method-card">
            <h4>🎯 GrabCut Algorithm</h4>
            <p><b>Best for:</b> All image types, especially complex ones</p>
            <p><b>Adjust rectangle</b> to define object area</p>
            </div>
            """, unsafe_allow_html=True)
            
            col_a, col_b = st.columns(2)
            with col_a:
                rect_x = st.slider("X Position", 0, img.shape[1], int(img.shape[1]*0.1))
                rect_y = st.slider("Y Position", 0, img.shape[0], int(img.shape[0]*0.1))
            with col_b:
                rect_w = st.slider("Width", 10, img.shape[1], int(img.shape[1]*0.8))
                rect_h = st.slider("Height", 10, img.shape[0], int(img.shape[0]*0.8))
                iterations = st.slider("Iterations", 1, 10, 5)
            
            rect = (rect_x, rect_y, rect_w, rect_h)
    
    # Tombol proses
    st.markdown("---")
    if st.button("🚀 " + ("Proses Sekarang!" if language == "Indonesia" else "Process Now!"), type="primary", use_container_width=True):
        
        with st.spinner("⏳ " + ("Memproses..." if language == "Indonesia" else "Processing...")):
            
            # Pilih metode berdasarkan pilihan
            if "GrabCut" in selected_method:
                result, rgba, mask = remove_background_grabcut(img, rect, iterations)
                method_name = "GrabCut"
                
            elif "Canny" in selected_method:
                result, rgba, mask = remove_background_canny_floodfill(img)
                method_name = "Canny Edge + Flood Fill"
                
            elif "HSV" in selected_method:
                result, rgba, mask, hsv_range = remove_background_improved_hsv(img)
                method_name = "Improved HSV"
                
            elif "Hybrid" in selected_method:
                result, rgba, mask = remove_background_hybrid(img)
                method_name = "Hybrid K-Means"
            
            # Hitung statistik
            total_pixels = mask.size
            foreground_pixels = np.sum(mask > 0)
            background_pixels = total_pixels - foreground_pixels
            foreground_percentage = (foreground_pixels / total_pixels) * 100
            
            # Tampilkan hasil
            st.markdown("---")
            if language == "Indonesia":
                st.subheader(f"🎨 Hasil dengan {method_name}")
            else:
                st.subheader(f"🎨 Results with {method_name}")
            
            # Tampilkan gambar hasil
            col_result1, col_result2, col_result3 = st.columns(3)
            
            with col_result1:
                st.image(mask, 
                        caption="🎭 " + ("Mask Deteksi" if language == "Indonesia" else "Detection Mask"), 
                        use_container_width=True,
                        clamp=True)
            
            with col_result2:
                st.image(result, 
                        caption="🖼️ " + ("Hasil Tanpa Background" if language == "Indonesia" else "Result without Background"), 
                        use_container_width=True)
            
            with col_result3:
                # Preview transparan dengan checkerboard
                h, w = result.shape[:2]
                checkerboard = create_checkerboard_background(h, w)
                alpha = rgba[:,:,3] / 255.0
                preview = np.zeros((h, w, 3), dtype=np.uint8)
                for c in range(3):
                    preview[:,:,c] = checkerboard[:,:,c] * (1 - alpha) + rgba[:,:,c] * alpha
                
                st.image(preview, 
                        caption="🔲 " + ("Preview Transparan" if language == "Indonesia" else "Transparent Preview"), 
                        use_container_width=True)
            
            # Statistik
            if language == "Indonesia":
                st.subheader("📊 Statistik Hasil")
            else:
                st.subheader("📊 Result Statistics")
            
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            with col_stat1:
                st.metric("Pixel Foreground", f"{foreground_pixels:,}", f"{foreground_percentage:.1f}%")
            with col_stat2:
                st.metric("Pixel Background", f"{background_pixels:,}", f"{(100-foreground_percentage):.1f}%")
            with col_stat3:
                st.metric("Total Pixel", f"{total_pixels:,}")
            
            # Rekomendasi
            if language == "Indonesia":
                if foreground_percentage < 10:
                    st.error("**❌ Objek terlalu kecil.** Coba metode lain atau perbesar rectangle")
                elif foreground_percentage > 90:
                    st.warning("**⚠️ Mungkin masih ada background.** Coba metode Canny Edge")
                else:
                    st.success("**✅ Hasil optimal!**")
            else:
                if foreground_percentage < 10:
                    st.error("**❌ Object too small.** Try different method or enlarge rectangle")
                elif foreground_percentage > 90:
                    st.warning("**⚠️ Possible background remaining.** Try Canny Edge method")
                else:
                    st.success("**✅ Optimal result!**")
            
            # Download buttons
            st.markdown("---")
            if language == "Indonesia":
                st.subheader("📥 Download Hasil")
            else:
                st.subheader("📥 Download Results")
            
            col_dl1, col_dl2, col_dl3 = st.columns(3)
            
            with col_dl1:
                result_bytes = cv2.imencode('.png', result)[1].tobytes()
                st.download_button(
                    label="📷 PNG (Hasil)",
                    data=result_bytes,
                    file_name=f"background_removed_{method_name.lower().replace(' ', '_')}.png",
                    mime="image/png",
                    use_container_width=True
                )
            
            with col_dl2:
                transparent_bytes = cv2.imencode('.png', rgba)[1].tobytes()
                st.download_button(
                    label="🔲 PNG (Transparan)",
                    data=transparent_bytes,
                    file_name=f"transparent_{method_name.lower().replace(' ', '_')}.png",
                    mime="image/png",
                    use_container_width=True
                )
            
            with col_dl3:
                mask_bytes = cv2.imencode('.png', mask)[1].tobytes()
                st.download_button(
                    label="🎭 PNG (Mask)",
                    data=mask_bytes,
                    file_name=f"mask_{method_name.lower().replace(' ', '_')}.png",
                    mime="image/png",
                    use_container_width=True
                )
            
            # Tips untuk metode lain
            st.markdown("---")
            if language == "Indonesia":
                st.info("""
                **🔄 Ingin Mencoba Metode Lain?**
                - **GrabCut**: Terbaik untuk semua jenis gambar
                - **Canny Edge**: Bagus untuk objek dengan tepi jelas
                - **Improved HSV**: Cocok untuk background warna solid
                - **Hybrid K-Means**: Untuk background kompleks/bertekstur
                """)
            else:
                st.info("""
                **🔄 Want to Try Other Methods?**
                - **GrabCut**: Best for all image types
                - **Canny Edge**: Good for objects with clear edges
                - **Improved HSV**: Suitable for solid color backgrounds
                - **Hybrid K-Means**: For complex/textured backgrounds
                """)

# ==================================================
#  PAGE 4 — TEAM (WITH PHOTOS)
# ==================================================
def page_team():
    if language == "Indonesia":
        st.title("👥 Tim Pengembang")

        st.subheader("📸 Anggota Tim")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.image("Falah.jpg", caption="Muhammad Nurul Falah", use_container_width=True)
            st.markdown("**Making App**")

        with col2:
            st.image("Rayyan.jpg", caption="Rayyan Hasan Rabbani", use_container_width=True)
            st.markdown("**Making App**")

        with col3:
            st.image("Tobias.jpg", caption="Tobias Dashiel Hapsoro", use_container_width=True)
            st.markdown("**Making Apps**")

        st.markdown("---")
        st.markdown("""
        ### **Fitur Aplikasi:**
        
        **1. 🛠️ Image Tools**
        - Transformasi matriks (translasi, scaling, rotasi)
        - Operasi konvolusi (blur, sharpen, edge detection)
        - Shearing dan refleksi
        
        **2. 🎭 Advanced Background Removal**
        - 4 metode berbeda untuk hasil optimal
        - Support berbagai jenis background
        - Output PNG transparan
        
        **3. 🌐 Multi-language Support**
        - Bahasa Indonesia & Inggris
        - Interface user-friendly
        - Panduan lengkap
        """)

        st.markdown("---")
        st.subheader("📚 Panduan Penggunaan")

        col_guide1, col_guide2 = st.columns(2)
        with col_guide1:
            st.markdown("""
            **Image Tools:**
            1. Pilih halaman Image Tools  
            2. Upload gambar  
            3. Pilih tool  
            4. Atur parameter  
            5. Download hasil  
            """)

        with col_guide2:
            st.markdown("""
            **Background Removal:**
            1. Pilih halaman Background Removal  
            2. Upload gambar  
            3. Pilih metode (GrabCut direkomendasikan)  
            4. Atur rectangle  
            5. Download hasil  
            """)

    else:
        st.title("👥 Development Team")

        st.subheader("📸 Team Members")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.image("Falah.jpg", caption="Muhammad Nurul Falah", use_container_width=True)
            st.markdown("**Algorithm Development & Integration**")

        with col2:
            st.image("Rayyan.jpg", caption="Rayyan Hasan Rabbani", use_container_width=True)
            st.markdown("**UI/UX Design & Image Processing**")

        with col3:
            st.image("Tobias.jpg", caption="Tobias Dashiel Hapsoro", use_container_width=True)
            st.markdown("**Testing & Optimization**")

        st.markdown("---")
        st.markdown("""
        ### **Application Features:**
        
        **1. 🛠️ Image Tools**
        - Matrix transformations
        - Convolution filters
        - Shearing & reflection
        
        **2. 🎭 Advanced Background Removal**
        - 4 different methods
        - Various background support
        - Transparent PNG output
        
        **3. 🌐 Multi-language Support**
        - Indonesian & English
        - User-friendly UI
        - Complete guide
        """)


# ==================================================
#  MAIN NAVIGATION
# ==================================================
pages = {
    "🏠 Home": page_home,
    "🛠️ Image Tools": page_image_tools,
    "🎭 Background Removal": page_background_removal,
    "👥 Team": page_team,
}

st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Pilih Halaman:" if language == "Indonesia" else "Select Page:",
    list(pages.keys())
)

# Menambahkan informasi di sidebar
st.sidebar.markdown("---")
if language == "Indonesia":
    st.sidebar.info("""
    **Tips:**
    - Untuk transformasi gambar: **Image Tools**
    - Untuk hapus background: **Background Removal**
    - Coba **GrabCut** untuk hasil terbaik
    """)
else:
    st.sidebar.info("""
    **Tips:**
    - For image transformations: **Image Tools**
    - For background removal: **Background Removal**
    - Try **GrabCut** for best results
    """)

pages[page]()