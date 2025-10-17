import re
import os
import hashlib
import mimetypes
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import email
from email import policy
from email.parser import BytesParser
from collections import Counter

# Machine Learning imports
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')


class MLSpamDetector:
    """Machine Learning-based spam detector with multiple models"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
        # Multiple ML models for ensemble prediction
        self.models = {
            'naive_bayes': MultinomialNB(alpha=0.1),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boost': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'logistic_regression': LogisticRegression(max_iter=1000, random_state=42)
        }
        
        self.is_trained = False
        self.feature_names = []
    
    def extract_features(self, text: str) -> Dict[str, float]:
        """Extract additional features from text"""
        features = {}
        
        # Length features
        features['char_count'] = len(text)
        features['word_count'] = len(text.split())
        features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text else 0
        
        # Punctuation features
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['caps_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        
        # Special character features
        features['dollar_sign_count'] = text.count('$')
        features['url_count'] = len(re.findall(r'https?://', text))
        features['email_count'] = len(re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text))
        
        # Numeric features
        features['number_count'] = len(re.findall(r'\d+', text))
        features['has_discount'] = 1 if re.search(r'\d+%\s*off', text, re.IGNORECASE) else 0
        
        return features
    
    def train(self, texts: List[str], labels: List[int]):
        """Train all ML models"""
        print("Training ML models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Vectorize text
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        # Train each model
        results = {}
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train_vec, y_train)
            y_pred = model.predict(X_test_vec)
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = accuracy
            print(f"{name} accuracy: {accuracy:.4f}")
        
        self.is_trained = True
        return results
    
    def predict(self, text: str) -> Dict:
        """Predict if text is spam using ensemble of models"""
        if not self.is_trained:
            return {
                'error': 'Models not trained yet',
                'is_spam': False,
                'confidence': 0.0
            }
        
        # Vectorize input
        text_vec = self.vectorizer.transform([text])
        
        # Get predictions from all models
        predictions = {}
        probabilities = {}
        
        for name, model in self.models.items():
            pred = model.predict(text_vec)[0]
            predictions[name] = pred
            
            # Get probability if available
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(text_vec)[0]
                probabilities[name] = {
                    'spam_prob': prob[1],
                    'ham_prob': prob[0]
                }
        
        # Ensemble voting
        spam_votes = sum(predictions.values())
        total_votes = len(predictions)
        is_spam = spam_votes > total_votes / 2
        confidence = spam_votes / total_votes
        
        return {
            'is_spam': bool(is_spam),
            'confidence': float(confidence),
            'model_predictions': predictions,
            'probabilities': probabilities,
            'votes': f"{spam_votes}/{total_votes}"
        }
    
    def get_feature_importance(self, top_n: int = 20) -> Dict:
        """Get most important features for spam detection"""
        if not self.is_trained:
            return {'error': 'Models not trained yet'}
        
        importance_data = {}
        
        # For Logistic Regression - get coefficients
        if 'logistic_regression' in self.models:
            lr_model = self.models['logistic_regression']
            coef = lr_model.coef_[0]
            top_indices = np.argsort(np.abs(coef))[-top_n:][::-1]
            importance_data['logistic_regression'] = [
                (self.feature_names[i], float(coef[i]))
                for i in top_indices
            ]
        
        # For Random Forest - get feature importances
        if 'random_forest' in self.models:
            rf_model = self.models['random_forest']
            importances = rf_model.feature_importances_
            top_indices = np.argsort(importances)[-top_n:][::-1]
            importance_data['random_forest'] = [
                (self.feature_names[i], float(importances[i]))
                for i in top_indices
            ]
        
        return importance_data
    
    def save_models(self, filepath: str = 'spam_detector_models.pkl'):
        """Save trained models to disk"""
        if not self.is_trained:
            print("No trained models to save")
            return False
        
        model_data = {
            'vectorizer': self.vectorizer,
            'models': self.models,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Models saved to {filepath}")
        return True
    
    def load_models(self, filepath: str = 'spam_detector_models.pkl'):
        """Load trained models from disk"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.vectorizer = model_data['vectorizer']
            self.models = model_data['models']
            self.feature_names = model_data['feature_names']
            self.is_trained = model_data['is_trained']
            print(f"Models loaded from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False


class SpamAndRiskDetector:
    """Enhanced spam and risk detector with ML capabilities"""
    
    def __init__(self):
        # Initialize ML detector
        self.ml_detector = MLSpamDetector()
        
        # Rule-based spam keywords with weights
        self.spam_keywords = {
            'urgent': 2, 'act now': 3, 'limited time': 2, 'click here': 3,
            'congratulations': 2, 'winner': 2, 'free': 2, 'cash': 2,
            'prize': 2, 'guarantee': 2, 'no obligation': 2, 'risk-free': 2,
            'double your': 3, 'earn money': 2, 'work from home': 2,
            'nigerian prince': 5, 'inheritance': 3, 'password': 2,
            'verify account': 3, 'suspended': 3, 'reactivate': 3,
            'bitcoin': 2, 'cryptocurrency': 2, 'investment opportunity': 3,
            'weight loss': 2, 'viagra': 3, 'pharmacy': 2, 'prescription': 2
        }
        
        # Dangerous file extensions
        self.risky_extensions = {
            '.exe', '.bat', '.cmd', '.com', '.pif', '.scr', '.vbs', '.js',
            '.jar', '.msi', '.app', '.deb', '.rpm', '.dmg', '.pkg',
            '.ps1', '.psm1', '.sh', '.bash', '.run', '.elf'
        }
        
        # Suspicious file extensions
        self.suspicious_extensions = {
            '.zip', '.rar', '.7z', '.tar', '.gz', '.iso',
            '.docm', '.xlsm', '.pptm', '.dotm', '.xltm'
        }
        
        self.malicious_hashes = set()
    
    def train_ml_models(self, training_data: List[Dict]):
        """Train ML models with labeled data"""
        texts = [item['text'] for item in training_data]
        labels = [item['label'] for item in training_data]
        return self.ml_detector.train(texts, labels)
    
    def analyze_email(self, email_path: str, use_ml: bool = True) -> Dict:
        """Analyze an email file with ML and rule-based detection"""
        try:
            with open(email_path, 'rb') as f:
                msg = BytesParser(policy=policy.default).parse(f)
            
            results = {
                'is_spam': False,
                'rule_based_score': 0,
                'ml_prediction': None,
                'risk_level': 'LOW',
                'reasons': [],
                'details': {}
            }
            
            # Extract email components
            subject = msg.get('subject', '') or ''
            sender = msg.get('from', '') or ''
            body = self._get_email_body(msg)
            
            combined_text = f"{subject} {body}".lower()
            
            # Rule-based detection
            spam_score = 0
            found_keywords = []
            
            for keyword, weight in self.spam_keywords.items():
                if keyword in combined_text:
                    spam_score += weight
                    found_keywords.append(keyword)
            
            results['rule_based_score'] = spam_score
            results['details']['found_keywords'] = found_keywords
            
            # Check for suspicious patterns
            if self._check_suspicious_links(body):
                spam_score += 5
                results['reasons'].append('Contains suspicious links')
            
            if self._check_phishing_patterns(sender, body):
                spam_score += 5
                results['reasons'].append('Potential phishing attempt')
            
            if self._check_urgency_tactics(combined_text):
                spam_score += 3
                results['reasons'].append('Uses urgency tactics')
            
            # ML-based detection
            if use_ml and self.ml_detector.is_trained:
                ml_result = self.ml_detector.predict(combined_text)
                results['ml_prediction'] = ml_result
                
                # Combine ML and rule-based results
                if ml_result['is_spam'] and ml_result['confidence'] > 0.7:
                    results['is_spam'] = True
                    results['reasons'].append(f"ML confidence: {ml_result['confidence']:.2%}")
                elif spam_score >= 10:
                    results['is_spam'] = True
            else:
                # Use only rule-based if ML not available
                if spam_score >= 10:
                    results['is_spam'] = True
            
            # Check attachments
            attachments = self._get_attachments(msg)
            if attachments:
                risky_attachments = self._check_risky_attachments(attachments)
                results['details']['attachments'] = risky_attachments
                if risky_attachments['risky_files']:
                    spam_score += 10
                    results['reasons'].append('Contains risky attachments')
            
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
            
        except Exception as e:
            return {'error': str(e)}
    
    def _get_email_body(self, msg) -> str:
        """Extract email body text"""
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    try:
                        body += part.get_payload(decode=True).decode('utf-8', errors='ignore')
                    except:
                        pass
        else:
            try:
                body = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
            except:
                pass
        return body
    
    def _check_suspicious_links(self, text: str) -> bool:
        """Check for suspicious URL patterns"""
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
    
    def _check_phishing_patterns(self, sender: str, body: str) -> bool:
        """Check for common phishing patterns"""
        phishing_indicators = [
            r'verify.*account', r'confirm.*identity', r'suspend.*account',
            r'unusual.*activity', r'click.*immediately', r'update.*payment',
            r're-?activate', r'security.*alert', r'locked.*account'
        ]
        
        combined = f"{sender} {body}".lower()
        for pattern in phishing_indicators:
            if re.search(pattern, combined):
                return True
        return False
    
    def _check_urgency_tactics(self, text: str) -> bool:
        """Check for urgency/pressure tactics"""
        urgency_patterns = [
            r'act (now|immediately|fast)', r'expires? (today|soon|tonight)',
            r'limited time', r'(hurry|rush)', r'last chance', r'don\'t (miss|wait)'
        ]
        
        for pattern in urgency_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def _get_attachments(self, msg) -> List[Dict]:
        """Extract attachment information"""
        attachments = []
        for part in msg.walk():
            if part.get_content_disposition() == 'attachment':
                filename = part.get_filename()
                if filename:
                    attachments.append({
                        'filename': filename,
                        'content': part.get_payload(decode=True)
                    })
        return attachments
    
    def _check_risky_attachments(self, attachments: List[Dict]) -> Dict:
        """Check attachments for risks"""
        results = {
            'total': len(attachments),
            'risky_files': [],
            'suspicious_files': [],
            'safe_files': []
        }
        
        for attachment in attachments:
            filename = attachment['filename']
            ext = Path(filename).suffix.lower()
            content = attachment.get('content')
            
            file_info = {'name': filename, 'extension': ext, 'risks': []}
            
            if ext in self.risky_extensions:
                file_info['risks'].append('Dangerous executable file')
                results['risky_files'].append(file_info)
            elif ext in self.suspicious_extensions:
                file_info['risks'].append('Potentially dangerous archive/macro file')
                results['suspicious_files'].append(file_info)
            else:
                if filename.count('.') > 1:
                    file_info['risks'].append('Double extension detected')
                    results['suspicious_files'].append(file_info)
                else:
                    results['safe_files'].append(file_info)
            
            if content:
                file_hash = hashlib.sha256(content).hexdigest()
                if file_hash in self.malicious_hashes:
                    file_info['risks'].append('Known malicious file hash')
                    if file_info not in results['risky_files']:
                        results['risky_files'].append(file_info)
        
        return results
    
    def scan_file(self, file_path: str) -> Dict:
        """Scan a single file for risks"""
        results = {
            'file': file_path,
            'is_risky': False,
            'risk_level': 'LOW',
            'risks': []
        }
        
        if not os.path.exists(file_path):
            results['error'] = 'File not found'
            return results
        
        filename = os.path.basename(file_path)
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
        
        try:
            with open(file_path, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
                results['sha256'] = file_hash
                
                if file_hash in self.malicious_hashes:
                    results['is_risky'] = True
                    results['risk_level'] = 'CRITICAL'
                    results['risks'].append('Known malicious file hash')
        except Exception as e:
            results['hash_error'] = str(e)
        
        file_size = os.path.getsize(file_path)
        results['size_bytes'] = file_size
        
        if ext in self.risky_extensions and file_size < 1024:
            results['risks'].append('Suspiciously small executable')
        
        return results


# Example usage and training data generator
def generate_sample_training_data() -> List[Dict]:
    """Generate sample training data for demonstration"""
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
        "Last chance! Premium membership expires today. Renew now or lose benefits!"
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
        "Weekly newsletter: Here are this week's top industry insights."
    ]
    
    training_data = []
    for text in spam_samples:
        training_data.append({'text': text, 'label': 1})  # 1 = spam
    for text in ham_samples:
        training_data.append({'text': text, 'label': 0})  # 0 = ham
    
    return training_data


if __name__ == "__main__":
    print("=== ML-Enhanced Spam and Risk Detector ===\n")
    
    # Initialize detector
    detector = SpamAndRiskDetector()
    
    # Generate and train with sample data
    print("Step 1: Training ML Models")
    print("-" * 50)
    training_data = generate_sample_training_data()
    results = detector.train_ml_models(training_data)
    print("\nTraining completed!")
    
    # Save models
    detector.ml_detector.save_models()
    
    print("\n" + "="*50 + "\n")
    
    # Example predictions
    print("Step 2: Testing Predictions")
    print("-" * 50)
    
    test_texts = [
        "Click here to claim your free prize! Act now before it expires!",
        "Let's schedule a meeting to discuss the project timeline.",
        "URGENT: Your account has been compromised. Verify your password immediately!"
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nTest {i}: '{text[:60]}...'")
        prediction = detector.ml_detector.predict(text)
        print(f"Prediction: {'SPAM' if prediction['is_spam'] else 'HAM'}")
        print(f"Confidence: {prediction['confidence']:.2%}")
    
    print("\n" + "="*50 + "\n")
    
    # Feature importance
    print("Step 3: Top Spam Indicators")
    print("-" * 50)
    importance = detector.ml_detector.get_feature_importance(top_n=10)
    if 'logistic_regression' in importance:
        print("\nTop spam keywords (Logistic Regression):")
        for word, score in importance['logistic_regression'][:5]:
            print(f"  - {word}: {score:.4f}")
    
    print("\n" + "="*50 + "\n")
    
    # Email analysis example
    print("Step 4: Email Analysis Example")
    print("-" * 50)
    email_path = "sample_email.eml"
    if os.path.exists(email_path):
        result = detector.analyze_email(email_path)
        print(f"Is Spam: {result['is_spam']}")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Rule-based Score: {result['rule_based_score']}")
        if result['ml_prediction']:
            print(f"ML Confidence: {result['ml_prediction']['confidence']:.2%}")
        print(f"Reasons: {', '.join(result['reasons'])}")
    else:
        print(f"Note: Place a .eml file at '{email_path}' to test email analysis")
    
    print("\n" + "="*50)
    print("\nML models trained and ready to use!")
    print("Models saved to 'spam_detector_models.pkl'")