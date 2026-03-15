import React, { useState, useEffect } from 'react';
import { register, checkEmailUsage } from '../api';
import { useNavigate, Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { UserPlus, Mail, Lock, User, Briefcase, ArrowRight, Loader2 } from 'lucide-react';

const Register = () => {
    const [username, setUsername] = useState('');
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [role, setRole] = useState('DEV');
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);
    const [emailCount, setEmailCount] = useState(0);
    const [emailLimit, setEmailLimit] = useState(10);
    const [checkingEmail, setCheckingEmail] = useState(false);
    const navigate = useNavigate();

    useEffect(() => {
        if (email.includes('@') && email.includes('.')) {
            const timer = setTimeout(async () => {
                setCheckingEmail(true);
                try {
                    const data = await checkEmailUsage(email);
                    setEmailCount(data.count);
                    setEmailLimit(data.limit);
                } catch (err) {
                    console.error("Failed to check email count", err);
                } finally {
                    setCheckingEmail(false);
                }
            }, 500);
            return () => clearTimeout(timer);
        } else {
            setEmailCount(0);
            setCheckingEmail(false);
        }
    }, [email]);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError('');
        try {
            await register(username, email, password, role);
            // Redirect to login with success message
            navigate('/login', { state: { message: 'Registration successful! Check your email for a welcome message.' } });
        } catch (err) {
            console.error("Registration error:", err);
            setError(err.response?.data?.detail || 'Registration failed. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="flex items-center justify-center min-h-screen overflow-hidden relative font-sans">
            {/* Floating Orbs Background */}
            <div className="floating-orb orb-1"></div>
            <div className="floating-orb orb-2"></div>
            <div className="floating-orb orb-3"></div>

            <motion.div
                initial={{ opacity: 0, y: 20, scale: 0.95 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                transition={{ duration: 0.6, ease: [0.4, 0, 0.2, 1] }}
                className="glass-card p-8 w-full max-w-md z-10 relative"
            >
                <div className="text-center mb-8">
                    <motion.div
                        initial={{ scale: 0.8, opacity: 0 }}
                        animate={{ scale: 1, opacity: 1 }}
                        transition={{ delay: 0.2, type: "spring", stiffness: 200 }}
                        className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-gradient-to-br from-indigo-500 to-purple-600 mb-4 shadow-lg glow-primary"
                    >
                        <UserPlus className="w-8 h-8 text-foreground" />
                    </motion.div>
                    <motion.h2
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.3 }}
                        className="text-3xl font-bold tracking-tight text-glow"
                    >
                        Create Account
                    </motion.h2>
                    <motion.p
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: 0.4 }}
                        className="text-muted mt-2"
                    >
                        Join the team and start collaborating
                    </motion.p>
                </div>

                {error && (
                    <motion.div
                        initial={{ opacity: 0, height: 0, scale: 0.95 }}
                        animate={{ opacity: 1, height: 'auto', scale: 1 }}
                        className="bg-red-500/10 border border-red-500/20 text-red-400 px-4 py-3 rounded-xl mb-6 text-sm text-center font-medium backdrop-blur-sm"
                    >
                        {error}
                    </motion.div>
                )}

                <form onSubmit={handleSubmit} className="space-y-4">
                    <motion.div
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: 0.3 }}
                        className="space-y-2"
                    >
                        <label className="text-sm font-medium text-muted ml-1">Username</label>
                        <div className="relative group">
                            <User className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-muted group-focus-within:text-indigo-400 transition-colors duration-300" />
                            <input
                                type="text"
                                value={username}
                                onChange={(e) => setUsername(e.target.value)}
                                className="input-with-icon"
                                placeholder="Choose a username"
                                required
                            />
                        </div>
                    </motion.div>

                    <motion.div
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: 0.4 }}
                        className="space-y-2"
                    >
                        <label className="text-sm font-medium text-muted ml-1">Email</label>
                        <div className="relative group">
                            <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-muted group-focus-within:text-indigo-400 transition-colors duration-300" />
                            <input
                                type="email"
                                value={email}
                                onChange={(e) => setEmail(e.target.value)}
                                className="input-with-icon"
                                placeholder="Enter your email"
                                required
                            />
                        </div>
                        {emailCount > 0 && (
                            <div className={`text-xs mt-1 ml-1 font-medium flex items-center gap-1 ${emailCount >= emailLimit ? 'text-red-400' : 'text-indigo-400'}`}>
                                {checkingEmail && <Loader2 className="w-3 h-3 animate-spin" />}
                                This email is used for {emailCount}/{emailLimit} accounts.
                                {emailCount >= emailLimit && ' Maximum reached.'}
                            </div>
                        )}
                    </motion.div>

                    <motion.div
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: 0.5 }}
                        className="space-y-2"
                    >
                        <label className="text-sm font-medium text-muted ml-1">Password</label>
                        <div className="relative group">
                            <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-muted group-focus-within:text-indigo-400 transition-colors duration-300" />
                            <input
                                type="password"
                                value={password}
                                onChange={(e) => setPassword(e.target.value)}
                                className="input-with-icon"
                                placeholder="Create a password"
                                required
                            />
                        </div>
                    </motion.div>

                    <motion.div
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: 0.6 }}
                        className="space-y-2"
                    >
                        <label className="text-sm font-medium text-muted ml-1">Role</label>
                        <div className="relative group">
                            <Briefcase className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-muted group-focus-within:text-indigo-400 transition-colors duration-300" />
                            <select
                                value={role}
                                onChange={(e) => setRole(e.target.value)}
                                className="input-with-icon appearance-none cursor-pointer bg-transparent"
                            >
                                <option value="DEV" className="bg-background">Developer</option>
                                <option value="TESTER" className="bg-background">Tester</option>
                                <option value="PO" className="bg-background">Product Owner</option>
                                <option value="PM" className="bg-background">Project Manager</option>
                                <option value="RE" className="bg-background">Requirement Engineer</option>
                            </select>
                        </div>
                    </motion.div>

                    <motion.button
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.7 }}
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                        type="submit"
                        disabled={loading}
                        className="w-full btn-primary py-3 disabled:opacity-70 disabled:cursor-not-allowed mt-4"
                    >
                        {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : <>Create Account <ArrowRight className="w-5 h-5" /></>}
                    </motion.button>
                </form>

                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.8 }}
                    className="mt-6 text-center"
                >
                    <p className="text-muted text-sm">
                        Already have an account?{' '}
                        <Link to="/login" className="text-indigo-400 hover:text-indigo-300 font-medium transition-colors hover:underline">
                            Sign In
                        </Link>
                    </p>
                </motion.div>
            </motion.div>
        </div>
    );
};

export default Register;
