#!/usr/bin/env python3
"""
=============================================================================
ALL-IN-ONE SPAM & RISK DETECTOR
Backend (Flask) + Frontend (React + Tailwind) in Single File
=============================================================================
Installation:
    pip install flask scikit-learn numpy flask-cors

Usage:
    python spam_detector_app.py
    
Then open: http://localhost:5000
=============================================================================
"""

import os
import re
import pickle
import hashlib
import warnings
import numpy as np
from pathlib import Path
from typing import Dict, List
from collections import Counter
import urllib.request
import urllib.parse
from urllib.error import URLError, HTTPError

# Flask imports
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ML imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')

# =============================================================================
# ML SPAM DETECTOR CLASS
# =============================================================================

class MLSpamDetector:
    """Machine Learning-based spam detector"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
        self.models = {
            'naive_bayes': MultinomialNB(alpha=0.1),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boost': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'logistic_regression': LogisticRegression(max_iter=1000, random_state=42)
        }
        
        self.is_trained = False
        self.feature_names = []
    
    def train(self, texts: List[str], labels: List[int]):
        """Train all ML models"""
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        results = {}
        for name, model in self.models.items():
            model.fit(X_train_vec, y_train)
            y_pred = model.predict(X_test_vec)
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = float(accuracy)
        
        self.is_trained = True
        return results
    
    def predict(self, text: str) -> Dict:
        """Predict if text is spam"""
        if not self.is_trained:
            return {
                'error': 'Models not trained yet',
                'is_spam': False,
                'confidence': 0.0
            }
        
        text_vec = self.vectorizer.transform([text])
        
        predictions = {}
        probabilities = {}
        
        for name, model in self.models.items():
            pred = model.predict(text_vec)[0]
            predictions[name] = int(pred)
            
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(text_vec)[0]
                probabilities[name] = {
                    'spam_prob': float(prob[1]),
                    'ham_prob': float(prob[0])
                }
        
        spam_votes = sum(predictions.values())
        total_votes = len(predictions)
        is_spam = spam_votes > total_votes / 2
        confidence = spam_votes / total_votes
        
        return {
            'is_spam': bool(is_spam),
            'confidence': float(confidence * 100),
            'model_predictions': predictions,
            'probabilities': probabilities,
            'votes': f"{spam_votes}/{total_votes}"
        }

# =============================================================================
# SPAM AND RISK DETECTOR CLASS
# =============================================================================

class SpamAndRiskDetector:
    """Enhanced spam and risk detector with ML capabilities"""
    
    def __init__(self):
        self.ml_detector = MLSpamDetector()
        
        self.spam_keywords = {
            'urgent': 2, 'act now': 3, 'limited time': 2, 'click here': 3,
            'congratulations': 2, 'winner': 2, 'free': 2, 'cash': 2,
            'prize': 2, 'guarantee': 2, 'no obligation': 2, 'risk-free': 2,
            'double your': 3, 'earn money': 2, 'work from home': 2,
            'nigerian prince': 5, 'inheritance': 3, 'password': 2,
            'verify account': 3, 'suspended': 3, 'reactivate': 3,
            'bitcoin': 2, 'cryptocurrency': 2, 'investment opportunity': 3,
            'weight loss': 2, 'viagra': 3, 'pharmacy': 2
        }
        
        self.risky_extensions = {
            '.exe', '.bat', '.cmd', '.com', '.pif', '.scr', '.vbs', '.js',
            '.jar', '.msi', '.app', '.deb', '.rpm', '.dmg', '.pkg',
            '.ps1', '.psm1', '.sh', '.bash', '.run', '.elf'
        }
        
        self.suspicious_extensions = {
            '.zip', '.rar', '.7z', '.tar', '.gz', '.iso',
            '.docm', '.xlsm', '.pptm', '.dotm', '.xltm'
        }
    
    def analyze_text(self, text: str, use_ml: bool = True) -> Dict:
        """Analyze text for spam"""
        results = {
            'is_spam': False,
            'rule_based_score': 0,
            'ml_prediction': None,
            'risk_level': 'LOW',
            'reasons': [],
            'details': {}
        }
        
        combined_text = text.lower()
        
        # Rule-based detection
        spam_score = 0
        found_keywords = []
        
        for keyword, weight in self.spam_keywords.items():
            if keyword in combined_text:
                spam_score += weight
                found_keywords.append(keyword)
        
        results['rule_based_score'] = spam_score
        results['details']['found_keywords'] = found_keywords
        
        # Check patterns
        if self._check_suspicious_links(text):
            spam_score += 5
            results['reasons'].append('Contains suspicious links')
        
        if self._check_phishing_patterns(text):
            spam_score += 5
            results['reasons'].append('Potential phishing attempt')
        
        if self._check_urgency_tactics(combined_text):
            spam_score += 3
            results['reasons'].append('Uses urgency tactics')
        
        # ML-based detection
        if use_ml and self.ml_detector.is_trained:
            ml_result = self.ml_detector.predict(text)
            results['ml_prediction'] = ml_result
            
            if ml_result['is_spam'] and ml_result['confidence'] > 70:
                results['is_spam'] = True
                results['reasons'].append(f"ML confidence: {ml_result['confidence']:.1f}%")
        
        if spam_score >= 10:
            results['is_spam'] = True
        
        # Determine risk level
        final_score = spam_score
        if results.get('ml_prediction') and results['ml_prediction']['is_spam']:
            final_score += 10
        
        if final_score >= 20:
            results['risk_level'] = 'HIGH'
        elif final_score >= 10:
            results['risk_level'] = 'MEDIUM'
        
        results['final_score'] = final_score
        
        return results
    
    def scan_file(self, filename: str) -> Dict:
        """Scan a file for risks"""
        results = {
            'file': filename,
            'is_risky': False,
            'risk_level': 'LOW',
            'risks': []
        }
        
        ext = Path(filename).suffix.lower()
        
        if ext in self.risky_extensions:
            results['is_risky'] = True
            results['risk_level'] = 'HIGH'
            results['risks'].append(f'Dangerous executable extension: {ext}')
        elif ext in self.suspicious_extensions:
            results['is_risky'] = True
            results['risk_level'] = 'MEDIUM'
            results['risks'].append(f'Suspicious file type: {ext}')
        
        if filename.count('.') > 1:
            results['risks'].append('Double extension detected')
            results['is_risky'] = True
            if results['risk_level'] == 'LOW':
                results['risk_level'] = 'MEDIUM'
        
        return results
    
    def analyze_email_link(self, url: str) -> Dict:
        """Analyze email content from a URL"""
        results = {
            'url': url,
            'accessible': False,
            'content_analysis': None,
            'url_analysis': {},
            'error': None
        }
        
        # Analyze URL structure first
        results['url_analysis'] = self._analyze_url_structure(url)
        
        try:
            # Try to fetch content
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=10) as response:
                content = response.read().decode('utf-8', errors='ignore')
                results['accessible'] = True
                
                # Analyze the fetched content
                results['content_analysis'] = self.analyze_text(content, use_ml=True)
                
        except (URLError, HTTPError, Exception) as e:
            results['error'] = str(e)
            results['accessible'] = False
        
        return results
    
    def _analyze_url_structure(self, url: str) -> Dict:
        """Analyze URL structure for suspicious patterns"""
        analysis = {
            'is_suspicious': False,
            'risk_level': 'LOW',
            'flags': []
        }
        
        url_lower = url.lower()
        
        # Check for suspicious domains
        suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.xyz', '.top', '.click']
        for tld in suspicious_tlds:
            if tld in url_lower:
                analysis['flags'].append(f'Suspicious TLD: {tld}')
                analysis['is_suspicious'] = True
        
        # Check for URL shorteners
        shorteners = ['bit.ly', 'tinyurl', 'goo.gl', 't.co', 'short.link']
        for shortener in shorteners:
            if shortener in url_lower:
                analysis['flags'].append(f'URL shortener detected: {shortener}')
                analysis['is_suspicious'] = True
        
        # Check for IP addresses
        ip_pattern = r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
        if re.search(ip_pattern, url):
            analysis['flags'].append('Direct IP address instead of domain')
            analysis['is_suspicious'] = True
        
        # Check for excessive subdomains
        domain_parts = url.split('/')[2].split('.')
        if len(domain_parts) > 4:
            analysis['flags'].append('Excessive subdomains detected')
            analysis['is_suspicious'] = True
        
        if analysis['is_suspicious']:
            analysis['risk_level'] = 'MEDIUM' if len(analysis['flags']) <= 2 else 'HIGH'
        
        return analysis
    
    def _check_suspicious_links(self, text: str) -> bool:
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, text)
        
        suspicious_patterns = [
            r'bit\.ly', r'tinyurl', r'goo\.gl',
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}',
            r'[a-z0-9-]+\.(xyz|top|club|work|click|online)',
        ]
        
        for url in urls:
            for pattern in suspicious_patterns:
                if re.search(pattern, url, re.IGNORECASE):
                    return True
        return False
    
    def _check_phishing_patterns(self, text: str) -> bool:
        phishing_indicators = [
            r'verify.*account', r'confirm.*identity', r'suspend.*account',
            r'unusual.*activity', r'click.*immediately', r'update.*payment',
            r're-?activate', r'security.*alert', r'locked.*account'
        ]
        
        combined = text.lower()
        for pattern in phishing_indicators:
            if re.search(pattern, combined):
                return True
        return False
    
    def _check_urgency_tactics(self, text: str) -> bool:
        urgency_patterns = [
            r'act (now|immediately|fast)', r'expires? (today|soon|tonight)',
            r'limited time', r'(hurry|rush)', r'last chance', r"don't (miss|wait)"
        ]
        
        for pattern in urgency_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

# =============================================================================
# TRAINING DATA
# =============================================================================

def generate_training_data() -> List[Dict]:
    """Generate training data"""
    spam_samples = [
        "URGENT! Your account will be suspended. Click here immediately to verify.",
        "Congratulations! You've won $1,000,000. Claim your prize now!",
        "Limited time offer! Buy viagra at 90% discount. Act fast!",
        "Work from home and earn $5000 per week. No experience needed!",
        "Your package is waiting. Confirm your address to receive inheritance.",
        "CLICK HERE NOW! Double your bitcoin investment guaranteed!",
        "Weight loss miracle! Lose 30 pounds in 7 days. Free trial!",
        "Your PayPal account has unusual activity. Verify immediately or lose access.",
        "Nigerian prince needs help. Transfer money and get 10 million reward.",
        "Last chance! Premium membership expires today. Renew now or lose benefits!",
        "Hot singles in your area want to meet you tonight!",
        "Click here for free money! No credit card required!",
        "Your parcel is waiting. Click to track: bit.ly/xyz123",
        "Earn $10,000 per week from home guaranteed!",
        "FINAL NOTICE: Your account will be closed. Act immediately!"
    ]
    
    ham_samples = [
        "Hi John, can we reschedule our meeting to next Tuesday at 3pm?",
        "Here's the quarterly report you requested. Let me know if you need clarification.",
        "Thank you for your order. Your package will arrive in 3-5 business days.",
        "Reminder: Team lunch tomorrow at noon in the conference room.",
        "Your subscription renewal is coming up next month. No action needed.",
        "Great meeting you at the conference. Looking forward to collaborating.",
        "Project update: We're on track to meet the Friday deadline.",
        "Happy birthday! Hope you have a wonderful day with family and friends.",
        "The documents you sent look good. I've signed and attached them.",
        "Weekly newsletter: Here are this week's top industry insights.",
        "Meeting notes from today's standup are now available.",
        "Can you review this document when you get a chance?",
        "Thanks for your help with the presentation yesterday.",
        "Lunch menu for next week has been posted.",
        "Your dentist appointment is scheduled for tomorrow at 2pm."
    ]
    
    training_data = []
    for text in spam_samples:
        training_data.append({'text': text, 'label': 1})
    for text in ham_samples:
        training_data.append({'text': text, 'label': 0})
    
    return training_data

# =============================================================================
# FLASK APP
# =============================================================================

app = Flask(__name__)
CORS(app)

# Initialize detector
detector = SpamAndRiskDetector()

# Train on startup
print("🎓 Training ML models...")
training_data = generate_training_data()
texts = [item['text'] for item in training_data]
labels = [item['label'] for item in training_data]
results = detector.ml_detector.train(texts, labels)
print("✅ Training completed!")
for model, acc in results.items():
    print(f"   • {model}: {acc:.2%}")

# =============================================================================
# REACT FRONTEND (EMBEDDED)
# =============================================================================

FRONTEND_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🛡️ Spam & Risk Detector</title>
    <script crossorigin src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
    <script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
    <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .animate-fade-in {
            animation: fadeIn 0.5s ease-out;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: .5; }
        }
        .animate-pulse-slow {
            animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
        }
    </style>
</head>
<body class="bg-gradient-to-br from-purple-600 via-purple-700 to-indigo-800 min-h-screen">
    <div id="root"></div>

    <script type="text/babel">
        const { useState } = React;

        function App() {
            const [activeTab, setActiveTab] = useState('analyze');
            const [textInput, setTextInput] = useState('');
            const [fileInput, setFileInput] = useState('');
            const [emailLinkInput, setEmailLinkInput] = useState('');
            const [uploadFile, setUploadFile] = useState(null);
            const [loading, setLoading] = useState(false);
            const [result, setResult] = useState(null);

            const analyzeText = async () => {
                if (!textInput.trim()) {
                    alert('Please enter some text to analyze');
                    return;
                }

                setLoading(true);
                setResult(null);

                try {
                    const response = await fetch('/api/analyze', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ text: textInput })
                    });
                    const data = await response.json();
                    setResult({ type: 'text', data });
                } catch (error) {
                    alert('Error analyzing text: ' + error.message);
                } finally {
                    setLoading(false);
                }
            };

            const scanFile = async () => {
                if (!fileInput.trim()) {
                    alert('Please enter a filename');
                    return;
                }

                setLoading(true);
                setResult(null);

                try {
                    const response = await fetch('/api/scan-file', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ filename: fileInput })
                    });
                    const data = await response.json();
                    setResult({ type: 'file', data });
                } catch (error) {
                    alert('Error scanning file: ' + error.message);
                } finally {
                    setLoading(false);
                }
            };

            const analyzeEmailLink = async () => {
                if (!emailLinkInput.trim()) {
                    alert('Please enter an email URL');
                    return;
                }

                setLoading(true);
                setResult(null);

                try {
                    const response = await fetch('/api/analyze-email-link', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ url: emailLinkInput })
                    });
                    const data = await response.json();
                    setResult({ type: 'email-link', data });
                } catch (error) {
                    alert('Error analyzing email link: ' + error.message);
                } finally {
                    setLoading(false);
                }
            };

            const uploadAndScan = async () => {
                if (!uploadFile) {
                    alert('Please select a file to upload');
                    return;
                }

                setLoading(true);
                setResult(null);

                try {
                    const formData = new FormData();
                    formData.append('file', uploadFile);
                    
                    const response = await fetch('/api/upload-file', {
                        method: 'POST',
                        body: formData
                    });
                    const data = await response.json();
                    setResult({ type: 'upload', data });
                } catch (error) {
                    alert('Error uploading file: ' + error.message);
                } finally {
                    setLoading(false);
                }
            };

            const ResultDisplay = ({ result }) => {
                if (!result) return null;

                if (result.type === 'text') {
                    const { data } = result;
                    const isSpam = data.is_spam;
                    const bgColor = isSpam ? 'bg-red-50 border-red-500' : 'bg-green-50 border-green-500';
                    const icon = isSpam ? '🚨' : '✅';
                    const title = isSpam ? 'SPAM DETECTED' : 'MESSAGE SAFE';

                    return (
                        <div className={`mt-6 p-6 rounded-xl border-l-4 ${bgColor} animate-fade-in`}>
                            <div className="flex items-center gap-3 mb-4">
                                <span className="text-4xl">{icon}</span>
                                <h3 className="text-2xl font-bold text-gray-800">{title}</h3>
                            </div>

                            <div className="space-y-4">
                                <div className="bg-white bg-opacity-60 p-4 rounded-lg">
                                    <p className="text-sm font-semibold text-gray-600 mb-2">Confidence Score:</p>
                                    <div className="w-full bg-gray-200 rounded-full h-6 overflow-hidden">
                                        <div 
                                            className={`h-full flex items-center justify-center text-white text-sm font-bold ${isSpam ? 'bg-red-500' : 'bg-green-500'}`}
                                            style={{ width: `${data.ml_prediction?.confidence || 50}%` }}
                                        >
                                            {Math.round(data.ml_prediction?.confidence || 50)}%
                                        </div>
                                    </div>
                                </div>

                                <div className="bg-white bg-opacity-60 p-4 rounded-lg">
                                    <p className="text-sm font-semibold text-gray-600 mb-2">Risk Level:</p>
                                    <span className={`inline-block px-4 py-2 rounded-full text-sm font-bold ${
                                        data.risk_level === 'HIGH' ? 'bg-red-500 text-white' :
                                        data.risk_level === 'MEDIUM' ? 'bg-yellow-500 text-white' :
                                        'bg-green-500 text-white'
                                    }`}>
                                        {data.risk_level}
                                    </span>
                                </div>

                                {data.reasons && data.reasons.length > 0 && (
                                    <div className="bg-white bg-opacity-60 p-4 rounded-lg">
                                        <p className="text-sm font-semibold text-gray-600 mb-2">Detection Reasons:</p>
                                        <ul className="list-disc list-inside space-y-1">
                                            {data.reasons.map((reason, i) => (
                                                <li key={i} className="text-gray-700">{reason}</li>
                                            ))}
                                        </ul>
                                    </div>
                                )}

                                {data.details?.found_keywords && data.details.found_keywords.length > 0 && (
                                    <div className="bg-white bg-opacity-60 p-4 rounded-lg">
                                        <p className="text-sm font-semibold text-gray-600 mb-2">Detected Keywords:</p>
                                        <div className="flex flex-wrap gap-2">
                                            {data.details.found_keywords.map((keyword, i) => (
                                                <span key={i} className="bg-orange-500 text-white px-3 py-1 rounded-full text-sm">
                                                    {keyword}
                                                </span>
                                            ))}
                                        </div>
                                    </div>
                                )}

                                {data.ml_prediction && (
                                    <div className="bg-white bg-opacity-60 p-4 rounded-lg">
                                        <p className="text-sm font-semibold text-gray-600 mb-2">ML Model Votes:</p>
                                        <p className="text-gray-700">{data.ml_prediction.votes} models detected spam</p>
                                    </div>
                                )}
                            </div>
                        </div>
                    );
                }

                if (result.type === 'email-link') {
                    const { data } = result;
                    const urlAnalysis = data.url_analysis;
                    const contentAnalysis = data.content_analysis;
                    const isAccessible = data.accessible;
                    
                    return (
                        <div className="mt-6 p-6 rounded-xl border-l-4 bg-blue-50 border-blue-500 animate-fade-in">
                            <div className="flex items-center gap-3 mb-4">
                                <span className="text-4xl">🔗</span>
                                <h3 className="text-2xl font-bold text-gray-800">EMAIL LINK ANALYSIS</h3>
                            </div>

                            <div className="space-y-4">
                                <div className="bg-white bg-opacity-60 p-4 rounded-lg">
                                    <p className="text-sm font-semibold text-gray-600 mb-1">URL:</p>
                                    <p className="text-gray-800 font-mono break-all">{data.url}</p>
                                </div>

                                <div className="bg-white bg-opacity-60 p-4 rounded-lg">
                                    <p className="text-sm font-semibold text-gray-600 mb-2">URL Risk Level:</p>
                                    <span className={`inline-block px-4 py-2 rounded-full text-sm font-bold ${
                                        urlAnalysis.risk_level === 'HIGH' ? 'bg-red-500 text-white' :
                                        urlAnalysis.risk_level === 'MEDIUM' ? 'bg-yellow-500 text-white' :
                                        'bg-green-500 text-white'
                                    }`}>
                                        {urlAnalysis.risk_level}
                                    </span>
                                </div>

                                {urlAnalysis.flags && urlAnalysis.flags.length > 0 && (
                                    <div className="bg-white bg-opacity-60 p-4 rounded-lg">
                                        <p className="text-sm font-semibold text-gray-600 mb-2">URL Flags:</p>
                                        <ul className="list-disc list-inside space-y-1">
                                            {urlAnalysis.flags.map((flag, i) => (
                                                <li key={i} className="text-gray-700">{flag}</li>
                                            ))}
                                        </ul>
                                    </div>
                                )}

                                <div className="bg-white bg-opacity-60 p-4 rounded-lg">
                                    <p className="text-sm font-semibold text-gray-600 mb-1">Content Accessible:</p>
                                    <p className={`font-semibold ${isAccessible ? 'text-green-600' : 'text-red-600'}`}>
                                        {isAccessible ? '✓ Yes' : '✗ No'}
                                    </p>
                                    {data.error && (
                                        <p className="text-red-600 text-sm mt-1">Error: {data.error}</p>
                                    )}
                                </div>

                                {contentAnalysis && (
                                    <div className="bg-white bg-opacity-60 p-4 rounded-lg">
                                        <p className="text-sm font-semibold text-gray-600 mb-2">Content Analysis:</p>
                                        <div className="space-y-2">
                                            <p className={`font-semibold ${contentAnalysis.is_spam ? 'text-red-600' : 'text-green-600'}`}>
                                                {contentAnalysis.is_spam ? '🚨 SPAM DETECTED' : '✅ CONTENT SAFE'}
                                            </p>
                                            <p className="text-sm">Risk Level: <span className="font-semibold">{contentAnalysis.risk_level}</span></p>
                                            {contentAnalysis.ml_prediction && (
                                                <p className="text-sm">ML Confidence: <span className="font-semibold">{Math.round(contentAnalysis.ml_prediction.confidence)}%</span></p>
                                            )}
                                        </div>
                                    </div>
                                )}
                            </div>
                        </div>
                    );
                }

                if (result.type === 'file') {
                    const { data } = result;
                    const isRisky = data.is_risky;
                    const bgColor = isRisky ? 'bg-yellow-50 border-yellow-500' : 'bg-green-50 border-green-500';
                    const icon = isRisky ? '⚠️' : '✅';
                    const title = isRisky ? 'RISKY FILE DETECTED' : 'FILE APPEARS SAFE';

                    return (
                        <div className={`mt-6 p-6 rounded-xl border-l-4 ${bgColor} animate-fade-in`}>
                            <div className="flex items-center gap-3 mb-4">
                                <span className="text-4xl">{icon}</span>
                                <h3 className="text-2xl font-bold text-gray-800">{title}</h3>
                            </div>

                            <div className="space-y-4">
                                <div className="bg-white bg-opacity-60 p-4 rounded-lg">
                                    <p className="text-sm font-semibold text-gray-600 mb-1">File Name:</p>
                                    <p className="text-gray-800 font-mono">{data.file}</p>
                                </div>

                                <div className="bg-white bg-opacity-60 p-4 rounded-lg">
                                    <p className="text-sm font-semibold text-gray-600 mb-2">Risk Level:</p>
                                    <span className={`inline-block px-4 py-2 rounded-full text-sm font-bold ${
                                        data.risk_level === 'HIGH' ? 'bg-red-500 text-white' :
                                        data.risk_level === 'MEDIUM' ? 'bg-yellow-500 text-white' :
                                        'bg-green-500 text-white'
                                    }`}>
                                        {data.risk_level}
                                    </span>
                                </div>

                                {data.risks && data.risks.length > 0 && (
                                    <div className="bg-white bg-opacity-60 p-4 rounded-lg">
                                        <p className="text-sm font-semibold text-gray-600 mb-2">Detected Risks:</p>
                                        <ul className="list-disc list-inside space-y-1">
                                            {data.risks.map((risk, i) => (
                                                <li key={i} className="text-gray-700">{risk}</li>
                                            ))}
                                        </ul>
                                    </div>
                                )}

                                <div className="bg-white bg-opacity-60 p-4 rounded-lg">
                                    <p className="text-sm font-semibold text-gray-600 mb-1">Recommendation:</p>
                                    <p className="text-gray-800 font-semibold">
                                        {isRisky 
                                            ? '🛑 Do not open this file unless from a trusted source. Scan with antivirus.' 
                                            : '✓ File type appears safe, but always verify source.'}
                                    </p>
                                </div>
                            </div>
                        </div>
                    );
                }

                if (result.type === 'upload') {
                    const { data } = result;
                    const fileAnalysis = data.file_analysis;
                    const contentAnalysis = data.content_analysis;
                    
                    return (
                        <div className="mt-6 p-6 rounded-xl border-l-4 bg-purple-50 border-purple-500 animate-fade-in">
                            <div className="flex items-center gap-3 mb-4">
                                <span className="text-4xl">📁</span>
                                <h3 className="text-2xl font-bold text-gray-800">FILE ANALYSIS COMPLETE</h3>
                            </div>

                            <div className="space-y-4">
                                <div className="bg-white bg-opacity-60 p-4 rounded-lg">
                                    <p className="text-sm font-semibold text-gray-600 mb-1">File Name:</p>
                                    <p className="text-gray-800 font-mono">{data.filename}</p>
                                    <p className="text-sm text-gray-600 mt-1">Size: {data.file_size} bytes</p>
                                </div>

                                <div className="bg-white bg-opacity-60 p-4 rounded-lg">
                                    <p className="text-sm font-semibold text-gray-600 mb-2">File Risk Assessment:</p>
                                    <span className={`inline-block px-4 py-2 rounded-full text-sm font-bold ${
                                        fileAnalysis.risk_level === 'HIGH' ? 'bg-red-500 text-white' :
                                        fileAnalysis.risk_level === 'MEDIUM' ? 'bg-yellow-500 text-white' :
                                        'bg-green-500 text-white'
                                    }`}>
                                        {fileAnalysis.risk_level}
                                    </span>
                                </div>

                                <div className="bg-white bg-opacity-60 p-4 rounded-lg">
                                    <p className="text-sm font-semibold text-gray-600 mb-2">Content Spam Analysis:</p>
                                    <div className="space-y-2">
                                        <p className={`font-semibold ${contentAnalysis.is_spam ? 'text-red-600' : 'text-green-600'}`}>
                                            {contentAnalysis.is_spam ? '🚨 SPAM DETECTED' : '✅ CONTENT SAFE'}
                                        </p>
                                        <p className="text-sm">Risk Level: <span className="font-semibold">{contentAnalysis.risk_level}</span></p>
                                        {contentAnalysis.ml_prediction && (
                                            <p className="text-sm">ML Confidence: <span className="font-semibold">{Math.round(contentAnalysis.ml_prediction.confidence)}%</span></p>
                                        )}
                                    </div>
                                </div>

                                {(fileAnalysis.risks && fileAnalysis.risks.length > 0) || (contentAnalysis.reasons && contentAnalysis.reasons.length > 0) && (
                                    <div className="bg-white bg-opacity-60 p-4 rounded-lg">
                                        <p className="text-sm font-semibold text-gray-600 mb-2">Detected Issues:</p>
                                        <ul className="list-disc list-inside space-y-1">
                                            {fileAnalysis.risks && fileAnalysis.risks.map((risk, i) => (
                                                <li key={`file-${i}`} className="text-gray-700">File: {risk}</li>
                                            ))}
                                            {contentAnalysis.reasons && contentAnalysis.reasons.map((reason, i) => (
                                                <li key={`content-${i}`} className="text-gray-700">Content: {reason}</li>
                                            ))}
                                        </ul>
                                    </div>
                                )}

                                <div className="bg-white bg-opacity-60 p-4 rounded-lg">
                                    <p className="text-sm font-semibold text-gray-600 mb-1">Overall Recommendation:</p>
                                    <p className="text-gray-800 font-semibold">
                                        {(fileAnalysis.is_risky || contentAnalysis.is_spam)
                                            ? '🛑 This file contains risky content or suspicious patterns. Exercise caution.' 
                                            : '✓ File appears safe based on analysis.'}
                                    </p>
                                </div>
                            </div>
                        </div>
                    );
                }

                return null;
            };

            return (
                <div className="container mx-auto px-4 py-8 max-w-4xl">
                    <header className="text-center text-white mb-8 animate-fade-in">
                        <h1 className="text-5xl font-bold mb-3">🛡️ Spam & Risk Detector</h1>
                        <p className="text-xl opacity-90">ML-Powered Email Security System</p>
                    </header>

                    <div className="bg-white rounded-2xl shadow-2xl overflow-hidden">
                        <div className="flex border-b">
                            <button
                                onClick={() => { setActiveTab('analyze'); setResult(null); }}
                                className={`flex-1 py-4 px-6 font-semibold transition-colors ${
                                    activeTab === 'analyze'
                                        ? 'bg-purple-600 text-white border-b-4 border-purple-800'
                                        : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                                }`}
                            >
                                📧 Analyze Text
                            </button>
                            <button
                                onClick={() => { setActiveTab('email-link'); setResult(null); }}
                                className={`flex-1 py-4 px-6 font-semibold transition-colors ${
                                    activeTab === 'email-link'
                                        ? 'bg-purple-600 text-white border-b-4 border-purple-800'
                                        : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                                }`}
                            >
                                🔗 Email Link
                            </button>
                            <button
                                onClick={() => { setActiveTab('file'); setResult(null); }}
                                className={`flex-1 py-4 px-6 font-semibold transition-colors ${
                                    activeTab === 'file'
                                        ? 'bg-purple-600 text-white border-b-4 border-purple-800'
                                        : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                                }`}
                            >
                                📁 File Scanner
                            </button>
                            <button
                                onClick={() => { setActiveTab('info'); setResult(null); }}
                                className={`flex-1 py-4 px-6 font-semibold transition-colors ${
                                    activeTab === 'info'
                                        ? 'bg-purple-600 text-white border-b-4 border-purple-800'
                                        : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                                }`}
                            >
                                ℹ️ Info
                            </button>
                        </div>

                        <div className="p-8">
                            {activeTab === 'analyze' && (
                                <div className="animate-fade-in">
                                    <label className="block text-gray-700 font-semibold mb-3">
                                        Enter Email or Message to Analyze:
                                    </label>
                                    <textarea
                                        value={textInput}
                                        onChange={(e) => setTextInput(e.target.value)}
                                        className="w-full h-40 px-4 py-3 border-2 border-gray-300 rounded-lg focus:border-purple-500 focus:outline-none transition-colors"
                                        placeholder="Paste email content, subject line, or message here..."
                                    />
                                    <button
                                        onClick={analyzeText}
                                        disabled={loading}
                                        className="mt-4 w-full bg-gradient-to-r from-purple-600 to-indigo-600 text-white font-bold py-4 rounded-lg hover:from-purple-700 hover:to-indigo-700 transition-all transform hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
                                    >
                                        {loading ? (
                                            <span className="flex items-center justify-center gap-2">
                                                <span className="animate-pulse-slow">🔍</span> Analyzing...
                                            </span>
                                        ) : (
                                            '🔍 Analyze for Spam'
                                        )}
                                    </button>
                                    <ResultDisplay result={result} />
                                </div>
                            )}

                            {activeTab === 'email-link' && (
                                <div className="animate-fade-in">
                                    <label className="block text-gray-700 font-semibold mb-3">
                                        Enter Email URL to Analyze:
                                    </label>
                                    <input
                                        type="url"
                                        value={emailLinkInput}
                                        onChange={(e) => setEmailLinkInput(e.target.value)}
                                        className="w-full px-4 py-3 border-2 border-gray-300 rounded-lg focus:border-purple-500 focus:outline-none transition-colors"
                                        placeholder="https://example.com/email-content"
                                    />
                                    <button
                                        onClick={analyzeEmailLink}
                                        disabled={loading}
                                        className="mt-4 w-full bg-gradient-to-r from-purple-600 to-indigo-600 text-white font-bold py-4 rounded-lg hover:from-purple-700 hover:to-indigo-700 transition-all transform hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
                                    >
                                        {loading ? (
                                            <span className="flex items-center justify-center gap-2">
                                                <span className="animate-pulse-slow">🔍</span> Analyzing...
                                            </span>
                                        ) : (
                                            '🔗 Analyze Email Link'
                                        )}
                                    </button>
                                    <ResultDisplay result={result} />
                                </div>
                            )}

                            {activeTab === 'file' && (
                                <div className="animate-fade-in space-y-6">
                                    <div className="bg-gray-50 p-4 rounded-lg">
                                        <h3 className="font-semibold text-gray-700 mb-3">📝 Check File Name Risk</h3>
                                        <input
                                            type="text"
                                            value={fileInput}
                                            onChange={(e) => setFileInput(e.target.value)}
                                            className="w-full px-4 py-3 border-2 border-gray-300 rounded-lg focus:border-purple-500 focus:outline-none transition-colors"
                                            placeholder="e.g., document.pdf, script.exe, invoice.zip"
                                        />
                                        <button
                                            onClick={scanFile}
                                            disabled={loading}
                                            className="mt-3 w-full bg-gradient-to-r from-blue-600 to-blue-700 text-white font-bold py-3 rounded-lg hover:from-blue-700 hover:to-blue-800 transition-all transform hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
                                        >
                                            {loading ? (
                                                <span className="flex items-center justify-center gap-2">
                                                    <span className="animate-pulse-slow">🔍</span> Scanning...
                                                </span>
                                            ) : (
                                                '🔍 Check File Name'
                                            )}
                                        </button>
                                    </div>

                                    <div className="text-center text-gray-500 font-semibold">OR</div>

                                    <div className="bg-gray-50 p-4 rounded-lg">
                                        <h3 className="font-semibold text-gray-700 mb-3">📤 Upload & Scan File Content</h3>
                                        <input
                                            type="file"
                                            onChange={(e) => setUploadFile(e.target.files[0])}
                                            className="w-full px-4 py-3 border-2 border-gray-300 rounded-lg focus:border-purple-500 focus:outline-none transition-colors"
                                            accept=".txt,.html,.eml,.msg,.json,.xml,.csv"
                                        />
                                        <p className="text-sm text-gray-600 mt-2">
                                            Supported: Text files, HTML, Email files (.eml), JSON, XML, CSV
                                        </p>
                                        <button
                                            onClick={uploadAndScan}
                                            disabled={loading || !uploadFile}
                                            className="mt-3 w-full bg-gradient-to-r from-purple-600 to-indigo-600 text-white font-bold py-3 rounded-lg hover:from-purple-700 hover:to-indigo-700 transition-all transform hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
                                        >
                                            {loading ? (
                                                <span className="flex items-center justify-center gap-2">
                                                    <span className="animate-pulse-slow">🔍</span> Scanning...
                                                </span>
                                            ) : (
                                                '📤 Upload & Scan File'
                                            )}
                                        </button>
                                    </div>
                                    
                                    <ResultDisplay result={result} />
                                </div>
                            )}

                            {activeTab === 'info' && (
                                <div className="animate-fade-in space-y-4">
                                    <div className="bg-blue-50 border-l-4 border-blue-500 p-4 rounded">
                                        <h3 className="font-bold text-blue-800 mb-2">🚀 How It Works</h3>
                                        <p className="text-gray-700">This system uses 4 machine learning models (Naive Bayes, Random Forest, Gradient Boosting, Logistic Regression) combined with rule-based detection to identify spam and risky files.</p>
                                    </div>

                                    <div className="bg-green-50 border-l-4 border-green-500 p-4 rounded">
                                        <h3 className="font-bold text-green-800 mb-2">✨ Features</h3>
                                        <ul className="list-disc list-inside space-y-1 text-gray-700">
                                            <li>ML ensemble voting for high accuracy</li>
                                            <li>Spam keyword detection</li>
                                            <li>Phishing pattern recognition</li>
                                            <li>Suspicious link analysis</li>
                                            <li>Email URL content analysis</li>
                                            <li>File name & content scanning</li>
                                            <li>Risky file extension detection</li>
                                            <li>Real-time confidence scoring</li>
                                        </ul>
                                    </div>

                                    <div className="bg-purple-50 border-l-4 border-purple-500 p-4 rounded">
                                        <h3 className="font-bold text-purple-800 mb-2">📊 Risk Levels</h3>
                                        <div className="space-y-2 text-gray-700">
                                            <p><span className="font-semibold text-green-600">🟢 LOW:</span> Safe, no concerns</p>
                                            <p><span className="font-semibold text-yellow-600">🟡 MEDIUM:</span> Suspicious, be cautious</p>
                                            <p><span className="font-semibold text-red-600">🔴 HIGH:</span> Dangerous, do not open/click</p>
                                        </div>
                                    </div>

                                    <div className="bg-orange-50 border-l-4 border-orange-500 p-4 rounded">
                                        <h3 className="font-bold text-orange-800 mb-2">🛠️ Technical Stack</h3>
                                        <ul className="list-disc list-inside space-y-1 text-gray-700">
                                            <li>Backend: Python + Flask</li>
                                            <li>ML: scikit-learn (TF-IDF, ensemble models)</li>
                                            <li>Frontend: React + Tailwind CSS</li>
                                            <li>All in one single Python file!</li>
                                        </ul>
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>

                    <footer className="text-center text-white mt-8 opacity-75">
                        <p>🛡️ Powered by Machine Learning • Built with React & Flask</p>
                    </footer>
                </div>
            );
        }

        ReactDOM.render(<App />, document.getElementById('root'));
    </script>
</body>
</html>
"""

# =============================================================================
# API ROUTES
# =============================================================================

@app.route('/')
def index():
    """Serve the React frontend"""
    return FRONTEND_HTML

@app.route('/api/analyze', methods=['POST'])
def analyze_text():
    """Analyze text for spam"""
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    result = detector.analyze_text(text, use_ml=True)
    return jsonify(result)

@app.route('/api/scan-file', methods=['POST'])
def scan_file():
    """Scan file for risks"""
    data = request.json
    filename = data.get('filename', '')
    
    if not filename:
        return jsonify({'error': 'No filename provided'}), 400
    
    result = detector.scan_file(filename)
    return jsonify(result)

@app.route('/api/analyze-email-link', methods=['POST'])
def analyze_email_link():
    """Analyze email content from URL"""
    data = request.json
    url = data.get('url', '')
    
    if not url:
        return jsonify({'error': 'No URL provided'}), 400
    
    result = detector.analyze_email_link(url)
    return jsonify(result)

@app.route('/api/upload-file', methods=['POST'])
def upload_file():
    """Upload and scan file content"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Read file content
        content = file.read().decode('utf-8', errors='ignore')
        
        # Analyze filename for risks
        file_analysis = detector.scan_file(file.filename)
        
        # Analyze content for spam
        content_analysis = detector.analyze_text(content, use_ml=True)
        
        result = {
            'filename': file.filename,
            'file_analysis': file_analysis,
            'content_analysis': content_analysis,
            'file_size': len(content)
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Error processing file: {str(e)}'}), 400

@app.route('/api/train', methods=['POST'])
def train_models():
    """Train models with custom data"""
    data = request.json
    training_data = data.get('training_data', [])
    
    if not training_data:
        return jsonify({'error': 'No training data provided'}), 400
    
    texts = [item['text'] for item in training_data]
    labels = [item['label'] for item in training_data]
    
    results = detector.ml_detector.train(texts, labels)
    return jsonify({'success': True, 'accuracies': results})

@app.route('/api/status', methods=['GET'])
def status():
    """Get system status"""
    return jsonify({
        'status': 'online',
        'ml_trained': detector.ml_detector.is_trained,
        'models': list(detector.ml_detector.models.keys())
    })

# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🛡️  SPAM & RISK DETECTOR - ALL-IN-ONE APPLICATION")
    print("="*70)
    print("\n✅ ML Models trained and ready!")
    print("\n🌐 Starting web server...")
    print("\n📱 Open your browser and go to:")
    print("   👉 http://localhost:5000")
    print("   👉 http://127.0.0.1:5000")
    print("\n💡 Press Ctrl+C to stop the server")
    print("="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)