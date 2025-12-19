"use client";

import { useState } from "react";
import { createClient } from "@/utils/supabase";
import { Zap, ArrowRight, Github, Mail } from "lucide-react";
import { motion } from "framer-motion";

export default function LoginPage() {
    const [email, setEmail] = useState("");
    const [loading, setLoading] = useState(false);
    const supabase = createClient();

    const handleLogin = async (e: React.FormEvent) => {
        e.preventDefault();
        setLoading(true);
        const { error } = await supabase.auth.signInWithOtp({
            email,
            options: {
                emailRedirectTo: process.env.NODE_ENV === "development"
                    ? "http://localhost:3000/auth/callback"
                    : "https://cursor-op.vercel.app/auth/callback",
            },
        });
        if (error) alert(error.message);
        else alert("Check your email for the login link!");
        setLoading(false);
    };

    return (
        <div style={{ minHeight: "100vh", display: "flex", alignItems: "center", justifyContent: "center", background: "var(--bg)" }}>
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="glass card"
                style={{ width: "100%", maxWidth: "400px", padding: "40px" }}
            >
                <div style={{ textAlign: "center", marginBottom: "40px" }}>
                    <div style={{ display: "inline-flex", background: "var(--accent)", padding: "12px", borderRadius: "8px", marginBottom: "20px" }}>
                        <Zap size={32} color="black" />
                    </div>
                    <h1 style={{ fontSize: "24px", fontWeight: 700, letterSpacing: "-0.04em" }}>Welcome to V1</h1>
                    <p className="text-muted" style={{ fontSize: "14px", marginTop: "8px" }}>The next generation LLM gateway.</p>
                </div>

                <form onSubmit={handleLogin} style={{ display: "grid", gap: "20px" }}>
                    <div>
                        <label className="stat-label" style={{ marginBottom: "8px", display: "block" }}>Work Email</label>
                        <input
                            type="email"
                            className="input"
                            placeholder="name@company.com"
                            style={{ width: "100%" }}
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            required
                        />
                    </div>
                    <button className="btn" style={{ width: "100%", padding: "12px" }} disabled={loading}>
                        {loading ? "Sending link..." : "Continue with Email"} <ArrowRight size={16} />
                    </button>
                </form>

                <div style={{ margin: "32px 0", display: "flex", alignItems: "center", gap: "12px" }}>
                    <div style={{ flex: 1, height: "1px", background: "var(--border)" }} />
                    <span className="text-muted" style={{ fontSize: "12px" }}>OR</span>
                    <div style={{ flex: 1, height: "1px", background: "var(--border)" }} />
                </div>

                <div style={{ display: "grid", gap: "12px" }}>
                    <button className="btn btn-outline" style={{ width: "100%", justifyContent: "center" }}>
                        <Github size={18} /> Continue with GitHub
                    </button>
                </div>

                <p className="text-muted" style={{ fontSize: "12px", textAlign: "center", marginTop: "32px" }}>
                    By continuing, you agree to our Terms of Service and Privacy Policy.
                </p>
            </motion.div>
        </div>
    );
}
