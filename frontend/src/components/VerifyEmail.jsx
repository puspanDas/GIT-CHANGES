import React, { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { verifyEmail } from '../api';
import { motion } from 'framer-motion';
import { CheckCircle, XCircle, Loader2, ArrowRight } from 'lucide-react';

const VerifyEmail = () => {
    const { token } = useParams();
    const [status, setStatus] = useState('verifying'); // 'verifying', 'success', 'error'
    const [message, setMessage] = useState('');

    useEffect(() => {
        const verify = async () => {
            try {
                const response = await verifyEmail(token);
                setMessage(response.message);
                setStatus('success');
            } catch (err) {
                setMessage(err.response?.data?.detail || 'Verification failed. The link may be invalid or expired.');
                setStatus('error');
            }
        };

        if (token) {
            verify();
        }
    }, [token]);

    return (
        <div className="flex items-center justify-center min-h-screen overflow-hidden relative font-sans">
            {/* Floating Orbs Background */}
            <div className="floating-orb orb-1"></div>
            <div className="floating-orb orb-2"></div>
            <div className="floating-orb orb-3"></div>

            <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                className="glass-card p-8 w-full max-w-md z-10 text-center"
            >
                {status === 'verifying' && (
                    <>
                        <motion.div
                            animate={{ rotate: 360 }}
                            transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                            className="inline-flex items-center justify-center w-20 h-20 rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 mb-6 shadow-lg"
                        >
                            <Loader2 className="w-10 h-10 text-foreground" />
                        </motion.div>
                        <h2 className="text-2xl font-bold mb-3 text-foreground">Verifying Your Email...</h2>
                        <p className="text-muted">Please wait while we verify your email address.</p>
                    </>
                )}

                {status === 'success' && (
                    <>
                        <motion.div
                            initial={{ scale: 0 }}
                            animate={{ scale: 1 }}
                            transition={{ type: "spring", stiffness: 200 }}
                            className="inline-flex items-center justify-center w-20 h-20 rounded-full bg-gradient-to-br from-green-500 to-emerald-600 mb-6 shadow-lg"
                        >
                            <CheckCircle className="w-10 h-10 text-foreground" />
                        </motion.div>
                        <h2 className="text-2xl font-bold mb-3 text-foreground">Email Verified! 🎉</h2>
                        <p className="text-muted mb-6">{message}</p>
                        <Link
                            to="/login"
                            className="inline-flex items-center gap-2 btn-primary py-3 px-6"
                        >
                            Continue to Login <ArrowRight className="w-4 h-4" />
                        </Link>
                    </>
                )}

                {status === 'error' && (
                    <>
                        <motion.div
                            initial={{ scale: 0 }}
                            animate={{ scale: 1 }}
                            transition={{ type: "spring", stiffness: 200 }}
                            className="inline-flex items-center justify-center w-20 h-20 rounded-full bg-gradient-to-br from-red-500 to-rose-600 mb-6 shadow-lg"
                        >
                            <XCircle className="w-10 h-10 text-foreground" />
                        </motion.div>
                        <h2 className="text-2xl font-bold mb-3 text-foreground">Verification Failed</h2>
                        <p className="text-muted mb-6">{message}</p>
                        <div className="space-y-3">
                            <Link to="/register" className="block w-full btn-secondary py-3">
                                Try Registering Again
                            </Link>
                            <Link to="/login" className="text-indigo-400 hover:text-indigo-300 text-sm font-medium">
                                ← Back to Login
                            </Link>
                        </div>
                    </>
                )}
            </motion.div>
        </div>
    );
};

export default VerifyEmail;
