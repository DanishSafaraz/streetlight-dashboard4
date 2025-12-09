import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import paho.mqtt.client as mqtt
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import socket
import joblib
import warnings
warnings.filterwarnings('ignore')

# ==================== KONFIGURASI MQTT ====================
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC_SENSOR = "iot/streetlight"
MQTT_TOPIC_CONTROL = "iot/streetlight/control"

# ==================== SESSION STATE INIT ====================
if "mqtt_connected" not in st.session_state:
    st.session_state.mqtt_connected = False

if "connection_status" not in st.session_state:
    st.session_state.connection_status = "❌ TIDAK TERKONEKSI"

if "connection_error" not in st.session_state:
    st.session_state.connection_error = ""

if "logs" not in st.session_state:
    st.session_state.logs = []

if "last_data" not in st.session_state:
    st.session_state.last_data = None

if "mqtt_client" not in st.session_state:
    st.session_state.mqtt_client = None

if "broker_test_result" not in st.session_state:
    st.session_state.broker_test_result = None

if "last_connection_attempt" not in st.session_state:
    st.session_state.last_connection_attempt = "Belum pernah"

if "debug_logs" not in st.session_state:
    st.session_state.debug_logs = []

# ==================== DEBUG LOGGING ====================
def add_debug_log(message):
    """Add debug message to logs"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    st.session_state.debug_logs.append(log_entry)
    # Keep only last 50 debug logs
    if len(st.session_state.debug_logs) > 50:
        st.session_state.debug_logs = st.session_state.debug_logs[-50:]
    print(log_entry)

# ==================== LOAD ML MODELS ====================
@st.cache_resource
def load_ml_models():
    """Load semua model ML"""
    models = {}
    add_debug_log("🔄 Memulai loading model ML...")
    
    try:
        # Load feature scaler
        models['scaler'] = joblib.load('feature_scaler.pkl')
        add_debug_log("✅ Feature Scaler loaded")
        st.success("✅ Feature Scaler loaded")
    except Exception as e:
        add_debug_log(f"⚠️ Feature Scaler not found: {e}")
        st.warning("⚠️ Feature Scaler not found")
        models['scaler'] = None
    
    try:
        # Load target encoder
        models['encoder'] = joblib.load('target_encoder.pkl')
        add_debug_log("✅ Target Encoder loaded")
        st.success("✅ Target Encoder loaded")
    except Exception as e:
        add_debug_log(f"⚠️ Target Encoder not found: {e}")
        st.warning("⚠️ Target Encoder not found")
        models['encoder'] = None
    
    try:
        # Load Decision Tree
        models['decision_tree'] = joblib.load('decision_tree.pkl')
        add_debug_log("✅ Decision Tree loaded")
        st.success("✅ Decision Tree loaded")
    except Exception as e:
        add_debug_log(f"⚠️ Decision Tree not found: {e}")
        st.warning("⚠️ Decision Tree not found")
        models['decision_tree'] = None
    
    try:
        # Load K-Nearest Neighbors
        models['knn'] = joblib.load('k-nearest_neighbors.pkl')
        add_debug_log("✅ K-Nearest Neighbors loaded")
        st.success("✅ K-Nearest Neighbors loaded")
    except Exception as e:
        add_debug_log(f"⚠️ K-Nearest Neighbors not found: {e}")
        st.warning("⚠️ K-Nearest Neighbors not found")
        models['knn'] = None
    
    try:
        # Load Logistic Regression
        models['logistic_regression'] = joblib.load('logistic_regression.pkl')
        add_debug_log("✅ Logistic Regression loaded")
        st.success("✅ Logistic Regression loaded")
    except Exception as e:
        add_debug_log(f"⚠️ Logistic Regression not found: {e}")
        st.warning("⚠️ Logistic Regression not found")
        models['logistic_regression'] = None
    
    # Cek jika semua model ada
    loaded_models = [k for k, v in models.items() if v is not None]
    add_debug_log(f"📊 Model yang berhasil di-load: {len(loaded_models)}/{len(models)}")
    
    return models

# Load models
ml_models = load_ml_models()

# ==================== FUNGSI PREDIKSI ML ====================
def make_predictions(intensity, voltage):
    """Membuat prediksi dari semua model ML"""
    predictions = {}
    add_debug_log(f"🔍 Memulai prediksi ML: Intensity={intensity}, Voltage={voltage}")
    
    # Jika tidak ada data input
    if intensity is None or voltage is None:
        add_debug_log("⚠️ Data input tidak lengkap untuk prediksi")
        return predictions
    
    try:
        # Pastikan values adalah float
        intensity_float = float(intensity)
        voltage_float = float(voltage)
        
        add_debug_log(f"📊 Data untuk ML: intensity={intensity_float}, voltage={voltage_float}")
        
        # Buat feature array - PERHATIAN: Periksa shape model Anda!
        # Biasanya model ML train dengan 2 features: [intensity, voltage]
        features = np.array([[intensity_float, voltage_float]])
        add_debug_log(f"📐 Features shape: {features.shape}")
        
        # Scale features jika scaler tersedia
        if ml_models['scaler'] is not None:
            try:
                features_scaled = ml_models['scaler'].transform(features)
                add_debug_log(f"⚖️ Features setelah scaling: {features_scaled}")
            except Exception as e:
                add_debug_log(f"❌ Error scaling features: {e}")
                features_scaled = features
        else:
            features_scaled = features
            add_debug_log("⚠️ Menggunakan features tanpa scaling")
        
        # Predict dengan Decision Tree
        if ml_models['decision_tree'] is not None:
            try:
                dt_pred = ml_models['decision_tree'].predict(features_scaled)[0]
                dt_prob = ml_models['decision_tree'].predict_proba(features_scaled)[0]
                predictions['Decision Tree'] = {
                    'prediction': dt_pred,
                    'confidence': float(np.max(dt_prob)),
                    'probabilities': dt_prob.tolist()
                }
                add_debug_log(f"🌳 Decision Tree: {dt_pred}, confidence: {np.max(dt_prob):.2%}")
            except Exception as e:
                add_debug_log(f"❌ Error Decision Tree prediction: {e}")
        
        # Predict dengan KNN
        if ml_models['knn'] is not None:
            try:
                knn_pred = ml_models['knn'].predict(features_scaled)[0]
                if hasattr(ml_models['knn'], 'predict_proba'):
                    knn_prob = ml_models['knn'].predict_proba(features_scaled)[0]
                    predictions['K-Nearest Neighbors'] = {
                        'prediction': knn_pred,
                        'confidence': float(np.max(knn_prob)),
                        'probabilities': knn_prob.tolist()
                    }
                else:
                    predictions['K-Nearest Neighbors'] = {
                        'prediction': knn_pred,
                        'confidence': None,
                        'probabilities': None
                    }
                add_debug_log(f"👥 KNN: {knn_pred}")
            except Exception as e:
                add_debug_log(f"❌ Error KNN prediction: {e}")
        
        # Predict dengan Logistic Regression
        if ml_models['logistic_regression'] is not None:
            try:
                lr_pred = ml_models['logistic_regression'].predict(features_scaled)[0]
                lr_prob = ml_models['logistic_regression'].predict_proba(features_scaled)[0]
                predictions['Logistic Regression'] = {
                    'prediction': lr_pred,
                    'confidence': float(np.max(lr_prob)),
                    'probabilities': lr_prob.tolist()
                }
                add_debug_log(f"📈 Logistic Regression: {lr_pred}, confidence: {np.max(lr_prob):.2%}")
            except Exception as e:
                add_debug_log(f"❌ Error Logistic Regression prediction: {e}")
        
        # Decode predictions jika encoder tersedia
        if ml_models['encoder'] is not None and predictions:
            try:
                for model_name, pred_data in predictions.items():
                    pred = pred_data['prediction']
                    if isinstance(pred, (int, np.integer)):
                        pred_decoded = ml_models['encoder'].inverse_transform([pred])[0]
                        pred_data['prediction_decoded'] = pred_decoded
                        add_debug_log(f"🔤 {model_name} decoded: {pred} -> {pred_decoded}")
                    else:
                        pred_data['prediction_decoded'] = pred
            except Exception as e:
                add_debug_log(f"❌ Error decoding predictions: {e}")
        
        # Voting dari semua model
        if predictions:
            all_predictions = []
            for model_name, pred_data in predictions.items():
                pred = pred_data.get('prediction_decoded', pred_data['prediction'])
                if pred is not None:
                    all_predictions.append(pred)
            
            if all_predictions:
                from collections import Counter
                vote_counts = Counter(all_predictions)
                majority_vote = vote_counts.most_common(1)[0][0]
                vote_confidence = vote_counts[majority_vote] / len(all_predictions)
                
                predictions['Ensemble Voting'] = {
                    'prediction': majority_vote,
                    'confidence': vote_confidence,
                    'votes': dict(vote_counts)
                }
                add_debug_log(f"🏆 Ensemble Voting: {majority_vote} (confidence: {vote_confidence:.2%})")
        
        add_debug_log(f"✅ Prediksi selesai: {len(predictions)} model berhasil")
        
    except Exception as e:
        add_debug_log(f"❌ Error dalam proses prediksi ML: {e}")
        st.error(f"ML Prediction Error: {e}")
    
    return predictions

# ==================== FUNGSI TEST KONEKSI ====================
def test_broker_connection():
    """Test connection to MQTT broker"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((MQTT_BROKER, MQTT_PORT))
        sock.close()
        return result == 0, None
    except Exception as e:
        return False, str(e)

# ==================== MQTT CALLBACKS ====================
def on_connect(client, userdata, flags, rc, properties=None):
    """Callback ketika koneksi MQTT berhasil/gagal"""
    if rc == 0:
        st.session_state.mqtt_connected = True
        st.session_state.connection_status = "✅ TERKONEKSI"
        st.session_state.connection_error = ""
        client.subscribe(MQTT_TOPIC_SENSOR)
        add_debug_log(f"✅ Connected to MQTT broker")
        add_debug_log(f"✅ Subscribed to topic: {MQTT_TOPIC_SENSOR}")
    else:
        st.session_state.mqtt_connected = False
        error_messages = {
            1: "Incorrect protocol version",
            2: "Invalid client identifier",
            3: "Server unavailable",
            4: "Bad username or password",
            5: "Not authorized"
        }
        error_msg = error_messages.get(rc, f"Error code: {rc}")
        st.session_state.connection_status = f"❌ {error_msg}"
        st.session_state.connection_error = error_msg
        add_debug_log(f"❌ Connection failed: {error_msg}")

def on_disconnect(client, userdata, rc):
    """Callback ketika terputus dari MQTT"""
    st.session_state.mqtt_connected = False
    st.session_state.connection_status = "❌ TERPUTUS"
    add_debug_log(f"⚠️ Disconnected from MQTT broker")

def on_message(client, userdata, msg):
    """Callback ketika menerima pesan MQTT"""
    try:
        payload = msg.payload.decode('utf-8', errors='ignore')
        add_debug_log(f"📥 Received MQTT message: {payload}")
        
        # Parse data dari ESP32 (format: {timestamp;intensity;voltage})
        if payload.startswith("{") and payload.endswith("}"):
            clean_payload = payload[1:-1]  # Remove curly braces
            parts = clean_payload.split(";")
            
            if len(parts) == 3:
                timestamp_str = parts[0].strip()
                intensity_str = parts[1].strip()
                voltage_str = parts[2].strip()
                
                add_debug_log(f"📝 Parsed parts: timestamp='{timestamp_str}', intensity='{intensity_str}', voltage='{voltage_str}'")
                
                # Parse values
                try:
                    intensity = float(intensity_str)
                except:
                    intensity = None
                    add_debug_log("⚠️ Failed to parse intensity as float")
                
                try:
                    voltage = float(voltage_str)
                except:
                    voltage = None
                    add_debug_log("⚠️ Failed to parse voltage as float")
                
                # Parse timestamp
                try:
                    timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                except:
                    timestamp = datetime.now()
                
                # Determine states berdasarkan logika ESP32
                if voltage == 0.0:
                    relay_state = "MATI"
                    lamp_state = "MENYALA"
                elif voltage == 220.0:
                    relay_state = "AKTIF"
                    lamp_state = "MATI"
                else:
                    relay_state = "UNKNOWN"
                    lamp_state = "UNKNOWN"
                
                add_debug_log(f"💡 Determined: Relay={relay_state}, Lamp={lamp_state}")
                
                # Make ML predictions
                predictions = make_predictions(intensity, voltage)
                
                # Get ensemble prediction if available
                ensemble_pred = None
                if 'Ensemble Voting' in predictions:
                    ensemble_pred = predictions['Ensemble Voting']['prediction']
                    add_debug_log(f"🏆 Ensemble prediction: {ensemble_pred}")
                
                # Create data row
                row = {
                    "timestamp": timestamp,
                    "intensity": intensity,
                    "voltage": voltage,
                    "relay_state": relay_state,
                    "lamp_state": lamp_state,
                    "ml_predictions": predictions,
                    "ensemble_prediction": ensemble_pred,
                    "source": "MQTT REAL"
                }
                
                # Update session state
                st.session_state.last_data = row
                st.session_state.logs.append(row)
                
                # Keep logs bounded
                if len(st.session_state.logs) > 1000:
                    st.session_state.logs = st.session_state.logs[-1000:]
                
                add_debug_log(f"✅ Data stored: Intensity={intensity}, Voltage={voltage}, Predictions={len(predictions)}")
                
            else:
                add_debug_log(f"⚠️ Invalid number of parts: {len(parts)} (expected 3)")
        else:
            add_debug_log(f"⚠️ Invalid payload format (missing curly braces): {payload}")
                
    except Exception as e:
        add_debug_log(f"❌ Error processing MQTT message: {e}")

# ==================== FUNGSI KONEKSI MQTT ====================
def connect_mqtt():
    """Connect to MQTT broker"""
    try:
        # Test broker connection first
        success, error = test_broker_connection()
        if not success:
            st.session_state.connection_status = f"❌ Broker tidak dapat diakses: {error}"
            st.session_state.connection_error = error
            add_debug_log(f"❌ Broker test failed: {error}")
            return False
        
        # Create MQTT client
        client = mqtt.Client()
        client.on_connect = on_connect
        client.on_message = on_message
        client.on_disconnect = on_disconnect
        
        # Connect
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        st.session_state.mqtt_client = client
        st.session_state.last_connection_attempt = datetime.now().strftime("%H:%M:%S")
        add_debug_log("✅ MQTT client created and connected")
        
        return True
        
    except Exception as e:
        st.session_state.connection_status = f"❌ Connection error: {str(e)}"
        st.session_state.connection_error = str(e)
        add_debug_log(f"❌ Connection failed: {e}")
        return False

def disconnect_mqtt():
    """Disconnect from MQTT broker"""
    if st.session_state.mqtt_client:
        try:
            st.session_state.mqtt_client.disconnect()
            add_debug_log("🔌 MQTT client disconnected")
        except:
            add_debug_log("⚠️ Error disconnecting MQTT client")
            pass
    st.session_state.mqtt_connected = False
    st.session_state.connection_status = "❌ TIDAK TERKONEKSI"
    st.session_state.mqtt_client = None

# ==================== STREAMLIT UI ====================
st.set_page_config(
    page_title="Smart Streetlight Dashboard with ML",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 SMART STREETLIGHT WITH ML PREDICTION")

# ==================== SIDEBAR ====================
with st.sidebar:
    st.title("⚙️ KONTROL SISTEM")
    
    # Status Connection
    st.subheader("🔗 STATUS KONEKSI")
    
    status_col1, status_col2 = st.columns([1, 3])
    with status_col1:
        if st.session_state.mqtt_connected:
            st.success("✅")
        else:
            st.error("❌")
    
    with status_col2:
        st.write(f"**Status:** {st.session_state.connection_status}")
        if st.session_state.connection_error:
            st.error(st.session_state.connection_error)
    
    st.write(f"**Terakhir dicoba:** {st.session_state.last_connection_attempt}")
    
    # Connection Buttons
    st.markdown("---")
    st.subheader("🔄 KONTROL MQTT")
    
    # Connect Button
    if st.button("🔗 Sambungkan ke MQTT", use_container_width=True, type="primary"):
        with st.spinner("Menghubungkan ke MQTT broker..."):
            if connect_mqtt():
                st.success("✅ Berhasil menghubungkan ke broker")
                st.session_state.broker_test_result = "✅ SUCCESS"
            else:
                st.error("❌ Gagal menghubungkan ke broker")
                st.session_state.broker_test_result = "❌ FAILED"
        time.sleep(1)
        st.rerun()
    
    # Disconnect Button
    if st.button("🔌 Putuskan Koneksi", use_container_width=True):
        disconnect_mqtt()
        st.warning("Koneksi MQTT diputuskan")
        time.sleep(1)
        st.rerun()
    
    # Test Connection Button
    if st.button("🧪 Test Koneksi Broker", use_container_width=True):
        with st.spinner("Testing koneksi ke broker..."):
            success, error = test_broker_connection()
            if success:
                st.success("✅ Broker dapat diakses dari server ini")
                st.session_state.broker_test_result = "✅ SUCCESS"
            else:
                st.error(f"❌ Broker tidak dapat diakses: {error}")
                st.session_state.broker_test_result = "❌ FAILED"
    
    # ML Model Status
    st.markdown("---")
    st.subheader("🤖 STATUS MODEL ML")
    
    model_status = {
        'Feature Scaler': ml_models.get('scaler') is not None,
        'Target Encoder': ml_models.get('encoder') is not None,
        'Decision Tree': ml_models.get('decision_tree') is not None,
        'K-Nearest Neighbors': ml_models.get('knn') is not None,
        'Logistic Regression': ml_models.get('logistic_regression') is not None
    }
    
    for model_name, status in model_status.items():
        if status:
            st.success(f"✅ {model_name}")
        else:
            st.error(f"❌ {model_name}")
    
    # Data Control
    st.markdown("---")
    st.subheader("📊 KONTROL DATA")
    
    if st.button("🗑️ Reset Data", use_container_width=True):
        st.session_state.logs = []
        st.session_state.last_data = None
        st.session_state.debug_logs = []
        st.success("Data telah direset")
        st.rerun()
    
    if st.button("🔄 Refresh Status", use_container_width=True):
        st.rerun()
    
    # Test Prediction Button
    st.markdown("---")
    st.subheader("🧪 TEST MANUAL")
    
    test_intensity = st.number_input("Test Intensity (%)", min_value=0.0, max_value=100.0, value=30.0)
    test_voltage = st.number_input("Test Voltage (V)", min_value=0.0, max_value=250.0, value=220.0)
    
    if st.button("🧪 Test Prediction", use_container_width=True):
        add_debug_log("🧪 Manual test prediction triggered")
        predictions = make_predictions(test_intensity, test_voltage)
        
        if predictions:
            st.success("✅ Test prediction successful!")
            for model_name, pred_data in predictions.items():
                pred = pred_data.get('prediction_decoded', pred_data.get('prediction', 'N/A'))
                confidence = pred_data.get('confidence')
                st.write(f"**{model_name}:** {pred} (confidence: {confidence:.2% if confidence else 'N/A'})")
        else:
            st.error("❌ No predictions generated")
    
    # Debug Panel Toggle
    show_debug = st.toggle("Show Debug Logs", value=False)

# ==================== MQTT LOOP POLLING ====================
# Process MQTT messages if connected
if st.session_state.mqtt_client:
    try:
        st.session_state.mqtt_client.loop(timeout=0.1)
    except Exception as e:
        add_debug_log(f"MQTT loop error: {e}")

# ==================== MAIN DASHBOARD ====================
# Status Banner
if not st.session_state.mqtt_connected:
    st.error("""
    ⚠️ **MQTT TIDAK TERKONEKSI!** 
    
    Silakan klik tombol "Sambungkan ke MQTT" di sidebar untuk menghubungkan ke broker.
    
    **Pastikan:**
    1. ESP32 menyala dan terhubung ke WiFi
    2. ESP32 mengirim data ke topic: `iot/streetlight`
    3. Format data: `{timestamp;intensity;voltage}`
    """)
else:
    if st.session_state.last_data:
        last_time = st.session_state.last_data.get('timestamp')
        if isinstance(last_time, datetime):
            time_str = last_time.strftime('%H:%M:%S')
        else:
            time_str = "N/A"
        st.success(f"✅ **TERHUBUNG KE MQTT BROKER** - Data terakhir: {time_str}")
    else:
        st.success("✅ **TERHUBUNG KE MQTT BROKER** - Menunggu data dari ESP32...")

# ==================== DEBUG PANEL ====================
if show_debug and st.session_state.debug_logs:
    with st.expander("🔍 DEBUG LOGS", expanded=True):
        for log in reversed(st.session_state.debug_logs[-20:]):  # Show last 20 logs
            st.text(log)
        
        if st.button("Clear Debug Logs"):
            st.session_state.debug_logs = []
            st.rerun()

# ==================== METRICS CARDS ====================
st.header("📊 STATUS REAL-TIME & PREDIKSI ML")

if st.session_state.last_data:
    data = st.session_state.last_data
    
    # Row 1: Sensor Data
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        intensity = data.get("intensity")
        if intensity is not None:
            if intensity < 30:
                color = "🟢"
                status_text = "GELAP"
                status_color = "normal"
            elif intensity < 70:
                color = "🟡"
                status_text = "SEDANG"
                status_color = "off"
            else:
                color = "🔵"
                status_text = "TERANG"
                status_color = "inverse"
            
            st.metric(
                label=f"{color} Intensitas Cahaya",
                value=f"{intensity:.1f}%",
                delta=f"{status_text}",
                delta_color=status_color
            )
        else:
            st.metric("Intensitas Cahaya", "N/A")
    
    with col2:
        voltage = data.get("voltage")
        relay_state = data.get("relay_state", "UNKNOWN")
        
        if relay_state == "AKTIF":
            icon = "🔴"
            bg_color = "#dc3545"
        elif relay_state == "MATI":
            icon = "🟢"
            bg_color = "#28a745"
        else:
            icon = "❓"
            bg_color = "#6c757d"
        
        st.markdown(f"""
        <div style="background-color: {bg_color}; padding: 15px; border-radius: 10px; color: white; text-align: center;">
            <div style="font-size: 14px;">{icon} Status Relay</div>
            <div style="font-size: 24px; font-weight: bold; margin: 5px 0;">{relay_state}</div>
            <div style="font-size: 16px;">{voltage if voltage is not None else 'N/A'} V</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        lamp_state = data.get("lamp_state", "UNKNOWN")
        
        if lamp_state == "MENYALA":
            icon = "💡"
            bg_color = "#FFD700"
            text_color = "black"
        elif lamp_state == "MATI":
            icon = "🌙"
            bg_color = "#2E4053"
            text_color = "white"
        else:
            icon = "❓"
            bg_color = "#6c757d"
            text_color = "white"
        
        st.markdown(f"""
        <div style="background-color: {bg_color}; padding: 15px; border-radius: 10px; color: {text_color}; text-align: center;">
            <div style="font-size: 14px;">Status Lampu</div>
            <div style="font-size: 36px; margin: 10px 0;">{icon}</div>
            <div style="font-size: 20px; font-weight: bold;">{lamp_state}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        ensemble_pred = data.get("ensemble_prediction")
        if ensemble_pred is not None:
            pred_str = str(ensemble_pred)
            if "MENYALA" in pred_str.upper() or "NYALA" in pred_str.upper() or "ON" in pred_str.upper():
                icon = "🤖💡"
                bg_color = "#4CAF50"
                pred_text = "REKOMENDASI: NYALA"
            elif "MATI" in pred_str.upper() or "OFF" in pred_str.upper():
                icon = "🤖🌙"
                bg_color = "#f44336"
                pred_text = "REKOMENDASI: MATI"
            else:
                icon = "🤖❓"
                bg_color = "#FF9800"
                pred_text = f"PREDIKSI: {ensemble_pred}"
        else:
            # Check individual predictions
            ml_predictions = data.get("ml_predictions", {})
            if ml_predictions:
                # Get first available prediction
                for model_name, pred_data in ml_predictions.items():
                    pred = pred_data.get('prediction_decoded', pred_data.get('prediction'))
                    if pred is not None:
                        pred_str = str(pred)
                        if "MENYALA" in pred_str.upper() or "NYALA" in pred_str.upper() or "ON" in pred_str.upper():
                            icon = "🤖💡"
                            bg_color = "#4CAF50"
                            pred_text = f"{model_name}: NYALA"
                            break
                        elif "MATI" in pred_str.upper() or "OFF" in pred_str.upper():
                            icon = "🤖🌙"
                            bg_color = "#f44336"
                            pred_text = f"{model_name}: MATI"
                            break
                else:
                    icon = "🤖⏳"
                    bg_color = "#9E9E9E"
                    pred_text = "Processing..."
            else:
                icon = "🤖⏳"
                bg_color = "#9E9E9E"
                pred_text = "Menunggu prediksi..."
        
        st.markdown(f"""
        <div style="background-color: {bg_color}; padding: 15px; border-radius: 10px; color: white; text-align: center;">
            <div style="font-size: 14px;">🤖 Prediksi ML</div>
            <div style="font-size: 36px; margin: 10px 0;">{icon}</div>
            <div style="font-size: 20px; font-weight: bold;">{pred_text}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Row 2: ML Predictions Details
    if data.get("ml_predictions"):
        st.markdown("---")
        st.subheader("📊 DETAIL PREDIKSI MODEL ML")
        
        predictions = data.get("ml_predictions", {})
        
        # Create columns for each model
        model_names = list(predictions.keys())
        if model_names:
            pred_cols = st.columns(len(model_names))
            
            for idx, model_name in enumerate(model_names):
                if idx < len(pred_cols):
                    pred_data = predictions[model_name]
                    with pred_cols[idx]:
                        pred = pred_data.get('prediction_decoded', pred_data.get('prediction', 'N/A'))
                        confidence = pred_data.get('confidence')
                        
                        # Determine color based on prediction
                        pred_str = str(pred).upper()
                        if "MENYALA" in pred_str or "NYALA" in pred_str or "ON" in pred_str:
                            pred_color = "#4CAF50"
                            pred_icon = "✅"
                        elif "MATI" in pred_str or "OFF" in pred_str:
                            pred_color = "#f44336"
                            pred_icon = "❌"
                        else:
                            pred_color = "#FF9800"
                            pred_icon = "⚠️"
                        
                        confidence_text = f"{confidence:.2%}" if confidence is not None else "N/A"
                        
                        st.markdown(f"""
                        <div style="background-color: {pred_color}; padding: 15px; border-radius: 10px; color: white; text-align: center; margin-bottom: 10px;">
                            <div style="font-size: 14px; font-weight: bold;">{model_name}</div>
                            <div style="font-size: 24px; margin: 10px 0;">{pred_icon} {pred}</div>
                            <div style="font-size: 14px;">Confidence: {confidence_text}</div>
                        </div>
                        """, unsafe_allow_html=True)
else:
    st.info("📭 **Belum ada data** - Tunggu data dari ESP32 atau sambungkan ke MQTT terlebih dahulu")

# ==================== VISUALISASI DATA ====================
st.header("📈 VISUALISASI DATA & PREDIKSI")

if st.session_state.logs:
    logs_list = st.session_state.logs[-200:]  # Last 200 points
    df = pd.DataFrame(logs_list)
    
    if not df.empty and "intensity" in df.columns:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Line chart dengan prediksi
            fig = go.Figure()
            
            # Plot intensity
            fig.add_trace(go.Scatter(
                x=df["timestamp"],
                y=df["intensity"],
                mode="lines+markers",
                name="Intensitas Cahaya",
                line=dict(color="#FFA500", width=3),
                marker=dict(size=8)
            ))
            
            # Add threshold line
            fig.add_hline(
                y=50,
                line_dash="dash",
                line_color="red",
                annotation_text="Threshold (50%)",
                annotation_position="bottom right"
            )
            
            fig.update_layout(
                title="TREN INTENSITAS CAHAYA",
                height=400,
                xaxis_title="Waktu",
                yaxis_title="Intensitas (%)",
                hovermode="x unified",
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Statistics
            if not df.empty:
                stats = {
                    "avg_intensity": df["intensity"].mean() if "intensity" in df.columns else 0,
                    "total_data": len(df),
                    "latest_time": df["timestamp"].max().strftime("%H:%M:%S") if "timestamp" in df.columns else "N/A"
                }
                
                st.metric("📊 Rata-rata Intensitas", f"{stats['avg_intensity']:.1f}%")
                st.metric("📈 Total Data", f"{stats['total_data']}")
                st.metric("🕐 Update Terakhir", stats['latest_time'])
                
                # ML Model Info
                st.markdown("---")
                st.subheader("🤖 INFO MODEL")
                
                loaded_count = sum(1 for v in ml_models.values() if v is not None)
                st.write(f"**Model loaded:** {loaded_count}/5")
                
                if st.session_state.last_data and st.session_state.last_data.get("ml_predictions"):
                    pred_count = len(st.session_state.last_data.get("ml_predictions", {}))
                    st.write(f"**Predictions made:** {pred_count}")
    else:
        st.warning("Data tidak lengkap untuk visualisasi")
else:
    st.info("📭 **Belum ada data untuk divisualisasikan**")

# ==================== TROUBLESHOOTING GUIDE ====================
with st.expander("🛠️ TROUBLESHOOTING GUIDE", expanded=False):
    col_t1, col_t2 = st.columns(2)
    
    with col_t1:
        st.markdown("""
        **🔧 ML Predictions Not Showing?**
        
        1. **Cek Format Data ESP32:**
           ```
           Format: {timestamp;intensity;voltage}
           Contoh: {2024-01-01 12:30:45;35;220.0}
           ```
        
        2. **Cek Model Files:**
           - Pastikan semua .pkl file ada di folder yang sama
           - File harus: `decision_tree.pkl`, `k-nearest_neighbors.pkl`, dll
        
        3. **Test Manual Prediction:**
           - Gunakan tombol "Test Prediction" di sidebar
           - Input intensity dan voltage manual
           - Lihat apakah prediksi muncul
        """)
    
    with col_t2:
        st.markdown("""
        **🔍 Debug Steps:**
        
        1. **Enable Debug Logs:**
           - Aktifkan "Show Debug Logs" di sidebar
           - Lihat pesan error yang muncul
        
        2. **Cek Data Format:**
           - Pastikan intensity adalah angka (0-100)
           - Pastikan voltage adalah angka (0.0 atau 220.0)
        
        3. **Test MQTT Manual:**
           - Buka: http://www.hivemq.com/demos/websocket-client/
           - Connect ke: broker.hivemq.com:1883
           - Subscribe ke: iot/streetlight
           - Lihat apakah data masuk dengan format benar
        """)
    
    # Quick Test Section
    st.markdown("---")
    st.subheader("🧪 QUICK TEST")
    
    test_col1, test_col2, test_col3 = st.columns(3)
    
    with test_col1:
        if st.button("Test Low Light (Lampu Nyala)"):
            predictions = make_predictions(25, 220.0)
            if predictions:
                st.success("✅ Test berhasil!")
                pred = list(predictions.values())[0].get('prediction_decoded', 'N/A')
                st.write(f"Prediksi: {pred}")
    
    with test_col2:
        if st.button("Test High Light (Lampu Mati)"):
            predictions = make_predictions(85, 0.0)
            if predictions:
                st.success("✅ Test berhasil!")
                pred = list(predictions.values())[0].get('prediction_decoded', 'N/A')
                st.write(f"Prediksi: {pred}")
    
    with test_col3:
        if st.button("Test Medium Light"):
            predictions = make_predictions(55, 0.0)
            if predictions:
                st.success("✅ Test berhasil!")
                pred = list(predictions.values())[0].get('prediction_decoded', 'N/A')
                st.write(f"Prediksi: {pred}")

# ==================== FOOTER ====================
st.divider()

footer_col1, footer_col2 = st.columns([1, 3])

with footer_col2:
    status_icon = "🟢" if st.session_state.mqtt_connected else "🔴"
    loaded_models = sum(1 for v in ml_models.values() if v is not None)
    
    st.markdown(f"""
    <div style="text-align: right; color: #666; font-size: 12px; padding: 10px;">
        <p>🤖 <strong>Smart Streetlight ML Dashboard</strong> | 
        MQTT: {status_icon} | ML Models: {loaded_models}/5 | 
        Data: {len(st.session_state.logs)} records | 
        Update: {datetime.now().strftime('%H:%M:%S')}</p>
    </div>
    """, unsafe_allow_html=True)

# CSS Styling
st.markdown("""
<style>
    /* Custom styling */
    div[data-testid="stMetricValue"] {
        font-size: 24px;
        font-weight: bold;
    }
    
    div[data-testid="stMetricDelta"] {
        font-size: 14px;
    }
    
    .stButton button {
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 3px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Auto-refresh
time.sleep(2)
st.rerun()
