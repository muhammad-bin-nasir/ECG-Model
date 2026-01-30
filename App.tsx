import React, { useState, useEffect, useRef } from 'react';
import {
  SafeAreaView,
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  PermissionsAndroid,
  Platform,
  Dimensions,
  Alert,
} from 'react-native';
import { BleManager, Device } from 'react-native-ble-plx';
import { Buffer } from 'buffer';
import Svg, { Path, Line } from 'react-native-svg';
import { InferenceSession, Tensor } from 'onnxruntime-react-native';
import RNFS from 'react-native-fs'; // Make sure to npm install react-native-fs

// =================================================================
// 🚀 CONTROL CENTER: TOGGLE THIS FOR YOUR DEMO
// =================================================================
const USE_MOCK_AI = true;  // <--- SET "FALSE" ONLY ON OLDER PHONE
// =================================================================

// --- CONFIGURATION ---
const SERVICE_UUID = '4fafc201-1fb5-459e-8fcc-c5c9c331914b';
const CHARACTERISTIC_UUID = 'beb5483e-36e1-4688-b7f5-ea07361b26a8';
const DEVICE_NAME = 'ECG-Guard';

// *** MODEL SETTINGS (MATCHING YOUR PYTHON TRAINING) ***
const FS = 250;                     // 250 Hz
const SEQ_LEN_SEC = 10;             // 10 Seconds
const MODEL_INPUT_SIZE = FS * SEQ_LEN_SEC; // 2500 Points
const ANOMALY_THRESHOLD = 0.30;     // Clinical Threshold

// Visual Settings
const SCREEN_WIDTH = Dimensions.get('window').width;
const GRAPH_HEIGHT = 200;
const VISUAL_POINTS = 50; // Points shown on graph

// --- MATH HELPERS (Replicating Python Logic) ---

const getMean = (data: number[]) => {
  const sum = data.reduce((a, b) => a + b, 0);
  return sum / data.length;
};

// Replicates scipy.signal.detrend(type='linear')
const detrend = (data: number[]) => {
  const n = data.length;
  if (n === 0) return data;

  let sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
  for (let i = 0; i < n; i++) {
    sumX += i;
    sumY += data[i];
    sumXY += i * data[i];
    sumXX += i * i;
  }

  const m = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
  const c = (sumY - m * sumX) / n;

  return data.map((val, i) => val - (m * i + c));
};

const App = () => {
  // --- STATE ---
  const [manager] = useState(new BleManager());
  const [device, setDevice] = useState<Device | null>(null);
  const [connectionStatus, setConnectionStatus] = useState('Disconnected');
  
  // AI State
  const [session, setSession] = useState<InferenceSession | null>(null);
  const [healthStatus, setHealthStatus] = useState('NORMAL'); 
  const [mseScore, setMseScore] = useState(0.0);
  
  // Visual Data State
  const [displayData, setDisplayData] = useState<number[]>(new Array(VISUAL_POINTS).fill(2048));
  const [currentVal, setCurrentVal] = useState<number>(0);
  
  // High-Speed Buffers
  const inferenceBuffer = useRef<number[]>([]); 

  // --- LIFECYCLE ---
  useEffect(() => {
    requestPermissions();
    initializeAI();
    return () => {
      manager.destroy();
    };
  }, []);

  // 1. Initialize AI (Mock or Real)
  const initializeAI = async () => {
    if (USE_MOCK_AI) {
      console.log("⚠️ MOCK MODE ENABLED: Skipping ONNX loading.");
      setSession({} as any); // Set a dummy session to avoid null checks
      return;
    }

    try {
      // REAL MODE: Copy model to cache to ensure readability
      const assetPath = 'ecg_model.onnx'; 
      const destPath = `${RNFS.DocumentDirectoryPath}/ecg_model.onnx`;

      if (!(await RNFS.exists(destPath))) {
        console.log("Copying model to cache...");
        if (Platform.OS === 'android') {
          await RNFS.copyFileAssets(assetPath, destPath);
        } else {
          await RNFS.copyFile(RNFS.MainBundlePath + '/' + assetPath, destPath);
        }
      }

      console.log("Loading model from:", destPath);
      const mySession = await InferenceSession.create(destPath);
      setSession(mySession);
      console.log("✅ AI Model Loaded Successfully!");

    } catch (e) {
      console.error("Model Load Error:", e);
      Alert.alert("Error", "Failed to load AI Model. Check logs.");
    }
  };

  const requestPermissions = async () => {
    if (Platform.OS === 'android') {
      await PermissionsAndroid.requestMultiple([
        PermissionsAndroid.PERMISSIONS.ACCESS_FINE_LOCATION,
        PermissionsAndroid.PERMISSIONS.BLUETOOTH_SCAN,
        PermissionsAndroid.PERMISSIONS.BLUETOOTH_CONNECT,
      ]);
    }
  };

  // --- BLUETOOTH LOGIC ---
  const scanAndConnect = () => {
    setConnectionStatus('Scanning...');
    manager.startDeviceScan(null, null, (error, scannedDevice) => {
      if (error) {
        setConnectionStatus('Scan Error');
        return;
      }
      if (scannedDevice && (scannedDevice.name === DEVICE_NAME || scannedDevice.localName === DEVICE_NAME)) {
        manager.stopDeviceScan();
        setConnectionStatus('Connecting...');
        connectToDevice(scannedDevice);
      }
    });
  };

  const connectToDevice = async (foundDevice: Device) => {
    try {
      const connectedDevice = await foundDevice.connect();
      setDevice(connectedDevice);
      await connectedDevice.discoverAllServicesAndCharacteristics();
      setConnectionStatus('Connected');
      startStreaming(connectedDevice);
    } catch (error) {
      setConnectionStatus('Connection Failed');
    }
  };

  const startStreaming = async (connectedDevice: Device) => {
    connectedDevice.monitorCharacteristicForService(
      SERVICE_UUID,
      CHARACTERISTIC_UUID,
      (error, characteristic) => {
        if (error) return;
        
        const rawValueBase64 = characteristic?.value;
        if (rawValueBase64) {
          const decodedString = Buffer.from(rawValueBase64, 'base64').toString('ascii');
          const value = parseInt(decodedString, 10);
          
          if (!isNaN(value)) {
            processNewDataPoint(value);
          }
        }
      }
    );
  };

  // --- CORE PIPELINE ---
  const processNewDataPoint = (value: number) => {
    setCurrentVal(value);
    
    // 1. Update Graph
    setDisplayData((prev) => {
      const newData = [...prev, value];
      if (newData.length > VISUAL_POINTS) newData.shift();
      return newData;
    });

    // 2. Add to AI Buffer
    inferenceBuffer.current.push(value);

    // 3. Check Trigger (2500 points = 10 seconds)
    if (inferenceBuffer.current.length >= MODEL_INPUT_SIZE) {
      
      const rawWindow = [...inferenceBuffer.current];
      
      // SLIDING LOGIC: Remove 1 second of data (FS = 250)
      inferenceBuffer.current = inferenceBuffer.current.slice(FS); 
      
      // Run AI
      performInference(rawWindow);
    }
  };

  const performInference = async (rawWindow: number[]) => {
    
    // --- STEP 1: PRE-PROCESSING (Common to both modes) ---
    // A. Detrend
    let processedData = detrend(rawWindow);
    // B. Mean Centering
    const mu = getMean(processedData);
    const normWindow = processedData.map(v => v - mu);


    // ==========================================
    // 🎭 MOCK MODE: SIMULATION
    // ==========================================
    if (USE_MOCK_AI) {
      // Simulate MSE based on signal variance.
      // If signal is "wild" (high variance), increase MSE.
      // If signal is "calm" (sine wave), low MSE.
      
      // Calculate variance of the normalized window
      const variance = normWindow.reduce((acc, val) => acc + (val * val), 0) / normWindow.length;
      
      // Heuristic: A clean sine wave usually has controlled variance. 
      // Random noise has spiky variance.
      // We'll generate a "Fake MSE" that hovers around 0.05 (Healthy)
      let simulatedMse = 0.05 + (Math.random() * 0.02);

      // Force Anomaly if variance is unnaturally high (just for demo logic)
      if (variance > 2000000) simulatedMse = 0.45; 

      setMseScore(simulatedMse);
      setHealthStatus(simulatedMse > ANOMALY_THRESHOLD ? 'ANOMALY' : 'NORMAL');
      return;
    }

    // ==========================================
    // 🧠 REAL MODE: ONNX INFERENCE
    // ==========================================
    if (!session) return;

    try {
      // Create Tensor [1, 2500, 1]
      const floatData = Float32Array.from(normWindow);
      const tensor = new Tensor('float32', floatData, [1, MODEL_INPUT_SIZE, 1]);

      // Run Model
      const feeds = { input: tensor }; 
      const results = await session.run(feeds);
      
      // Get Output
      const outputTensor = results[Object.keys(results)[0]];
      const reconstruction = outputTensor.data as Float32Array;

      // Calculate MSE
      let sumError = 0;
      for (let i = 0; i < floatData.length; i++) {
        const diff = floatData[i] - reconstruction[i];
        sumError += diff * diff;
      }
      const mse = sumError / floatData.length;

      // Update UI
      setMseScore(mse);
      setHealthStatus(mse > ANOMALY_THRESHOLD ? 'ANOMALY' : 'NORMAL');

    } catch (e) {
      console.log("Inference Error:", e);
    }
  };

  // --- VISUALIZATION ---
  const generatePath = () => {
    const xStep = SCREEN_WIDTH / (VISUAL_POINTS - 1);
    const min = Math.min(...displayData);
    const max = Math.max(...displayData);
    const range = max - min || 1; 

    const scaleY = (value: number) => {
      const norm = (value - min) / range;
      return GRAPH_HEIGHT - (norm * GRAPH_HEIGHT); 
    };

    let path = `M 0 ${scaleY(displayData[0])}`; 
    for (let i = 1; i < displayData.length; i++) {
      path += ` L ${i * xStep} ${scaleY(displayData[i])}`;
    }
    return path;
  };

  return (
    <SafeAreaView style={[styles.container, healthStatus === 'ANOMALY' ? styles.dangerBg : null]}>
      <View style={styles.header}>
        <Text style={styles.title}>ECG Guard</Text>
        <Text style={styles.status}>Status: {connectionStatus}</Text>
        {USE_MOCK_AI && <Text style={styles.mockTag}>[DEMO MODE]</Text>}
      </View>

      {/* --- GRAPH --- */}
      <View style={styles.graphContainer}>
        <Svg height={GRAPH_HEIGHT} width={SCREEN_WIDTH}>
          <Line x1="0" y1={GRAPH_HEIGHT/2} x2={SCREEN_WIDTH} y2={GRAPH_HEIGHT/2} stroke="#333" strokeWidth="1" />
          <Path d={generatePath()} fill="none" stroke="#fff" strokeWidth="2" />
        </Svg>
      </View>

      {/* --- AI DIAGNOSTICS --- */}
      <View style={styles.infoBox}>
        <Text style={styles.label}>AI Diagnosis:</Text>
        <Text style={[styles.bigStatus, healthStatus === 'NORMAL' ? styles.green : styles.red]}>
          {healthStatus}
        </Text>
        
        <Text style={styles.label}>Reconstruction Error (MSE):</Text>
        <Text style={styles.dataValue}>{mseScore.toFixed(4)}</Text>
        <Text style={styles.smallNote}>Threshold: {ANOMALY_THRESHOLD}</Text>
      </View>

      <View style={styles.dataContainer}>
         <Text style={styles.label}>Live Sensor Value:</Text>
         <Text style={styles.sensorValue}>{currentVal}</Text>
      </View>

      {!device && (
        <TouchableOpacity style={styles.button} onPress={scanAndConnect}>
          <Text style={styles.buttonText}>Scan & Connect</Text>
        </TouchableOpacity>
      )}

      {device && (
        <TouchableOpacity 
          style={[styles.button, styles.disconnectBtn]} 
          onPress={async () => {
            if (device) await device.cancelConnection();
            setDevice(null);
            setConnectionStatus('Disconnected');
          }}
        >
          <Text style={styles.buttonText}>Disconnect</Text>
        </TouchableOpacity>
      )}
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#111', paddingTop: 40 },
  dangerBg: { backgroundColor: '#330000' }, 
  header: { alignItems: 'center', marginBottom: 20 },
  title: { fontSize: 28, color: '#fff', fontWeight: 'bold' },
  status: { fontSize: 14, color: '#888', marginTop: 5 },
  mockTag: { color: 'yellow', fontSize: 12, fontWeight: 'bold', marginTop: 2},
  
  graphContainer: { 
    height: 200, 
    width: '100%', 
    backgroundColor: '#000', 
    borderTopWidth: 1, borderBottomWidth: 1, borderColor: '#333',
    justifyContent: 'center'
  },
  
  infoBox: { alignItems: 'center', marginTop: 20, padding: 20 },
  label: { color: '#aaa', fontSize: 16, marginTop: 5 },
  bigStatus: { fontSize: 32, fontWeight: 'bold', marginVertical: 5 },
  dataValue: { fontSize: 24, color: '#fff', fontWeight: 'bold' },
  smallNote: { color: '#555', fontSize: 12 },

  dataContainer: { alignItems: 'center', marginTop: 10},
  sensorValue: { fontSize: 20, color: '#888'},
  
  green: { color: '#00ffcc' },
  red: { color: '#ff4444' },
  
  button: {
    backgroundColor: '#00ffcc',
    padding: 15, marginHorizontal: 40, borderRadius: 10,
    alignItems: 'center', marginTop: 20,
  },
  disconnectBtn: { backgroundColor: '#ff4444' },
  buttonText: { color: '#000', fontSize: 18, fontWeight: 'bold' },
});

export default App;