"use client";

import { useEffect, useState, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
    ChevronDown, ArrowRight, MessageSquare, Clock,
    Terminal, Sparkles, Layout, ExternalLink, Search
} from "lucide-react";
import { useRouter } from "next/navigation";
import { Sidebar } from "@/components/Sidebar";
import { HeroSphere } from "@/components/HeroSphere";
import { createClient } from "@/utils/supabase";

// --- Types ---
interface Conversation {
    id: string;
    session_id: string;
    title: string;
    last_request_at: number;
}

interface AnalyticsRow {
    id: string;
    conversation_id: string;
    model: string;
    tokens_in: number;
    tokens_out: number;
    tokens_saved: number;
    latency_ms: number;
    timestamp: number;
    raw_messages: any[];
    response_message: any;
    reconstruction_log: {
        sequence: Array<{
            source: string;
            text: string;
            score: number;
            line_index: number;
        }>;
    };
}

export default function ObservabilityPage() {
    const router = useRouter();
    const [conversations, setConversations] = useState<Conversation[]>([]);
    const [selectedConvo, setSelectedConvo] = useState<string>("");
    const [requests, setRequests] = useState<AnalyticsRow[]>([]);
    const [selectedRequest, setSelectedRequest] = useState<AnalyticsRow | null>(null);
    const [loading, setLoading] = useState(true);

    const GATEWAY_URL = process.env.NEXT_PUBLIC_GATEWAY_URL || "http://localhost:8000";
    const supabase = createClient();

    useEffect(() => {
        const fetchStats = async () => {
            const { data: { user } } = await supabase.auth.getUser();
            if (user) {
                try {
                    const resp = await fetch(`${GATEWAY_URL}/api/user/stats`, {
                        headers: { "x-user-id": user.id }
                    });
                    const data = await resp.json();
                    setConversations(data.recent_conversations || []);
                } catch (e) {
                    console.error("Failed to fetch conversations", e);
                }
                setLoading(false);
            }
        };
        fetchStats();
    }, [supabase.auth]);

    const handleConvoChange = async (id: string) => {
        setSelectedConvo(id);
        setSelectedRequest(null);
        if (!id) return;

        try {
            const resp = await fetch(`${GATEWAY_URL}/api/conversations/${id}`);
            const data = await resp.json();
            setRequests(data.requests || []);
            // Auto-select the last request in the convo
            if (data.requests?.length > 0) {
                setSelectedRequest(data.requests[data.requests.length - 1]);
            }
        } catch (e) {
            console.error("Failed to fetch conversation details", e);
        }
    };

    return (
        <div className="app-container">
            <Sidebar />
            <main className="main-content" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', paddingTop: '60px', gap: '20px' }}>

                {/* Sphere Above Title */}
                <motion.div
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ duration: 1.5, ease: "easeOut" }}
                    style={{ width: '100%', maxWidth: '400px', marginBottom: '-40px' }}
                >
                    <HeroSphere />
                </motion.div>

                {/* Minimalist Title Section */}
                <div style={{ textAlign: 'center', zIndex: 10 }}>
                    <motion.h1
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        style={{
                            fontSize: '42px',
                            fontWeight: 800,
                            letterSpacing: '-0.04em',
                            marginBottom: '12px',
                            background: 'linear-gradient(to bottom, #fff, rgba(255,255,255,0.5))',
                            WebkitBackgroundClip: 'text',
                            WebkitTextFillColor: 'transparent'
                        }}
                    >
                        Deep Observability
                    </motion.h1>
                    <motion.p
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 0.4 }}
                        style={{ fontSize: '15px', fontWeight: 500 }}
                    >
                        Select a session to analyze context reconstruction
                    </motion.p>
                </div>

                {/* Simple Dropdown Selection */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.2 }}
                    style={{ width: '100%', maxWidth: '440px', display: 'flex', flexDirection: 'column', gap: '12px', zIndex: 10 }}
                >
                    <div className="custom-select-wrapper">
                        <select
                            value={selectedConvo}
                            onChange={(e) => handleConvoChange(e.target.value)}
                            className="glass custom-select"
                        >
                            <option value="" disabled>─ Select Conversation ─</option>
                            {conversations.map(c => (
                                <option key={c.id} value={c.id}>
                                    {c.title} ({new Date(c.last_request_at * 1000).toLocaleDateString()})
                                </option>
                            ))}
                        </select>
                        <ChevronDown className="select-icon" size={16} />
                    </div>

                    {requests.length > 0 && (
                        <div className="custom-select-wrapper">
                            <select
                                value={selectedRequest?.id || ""}
                                onChange={(e) => setSelectedRequest(requests.find(r => r.id === e.target.value) || null)}
                                className="glass custom-select"
                            >
                                {requests.map((r, i) => (
                                    <option key={r.id} value={r.id}>
                                        Request #{i + 1} • {r.model} • {new Date(r.timestamp * 1000).toLocaleTimeString()}
                                    </option>
                                ))}
                            </select>
                            <ChevronDown className="select-icon" size={16} />
                        </div>
                    )}
                </motion.div>

                {/* Preview Section */}
                <div style={{ width: '100%', maxWidth: '800px', marginTop: '20px' }}>
                    <AnimatePresence mode="wait">
                        {selectedRequest ? (
                            <motion.div
                                key={selectedRequest.id}
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                exit={{ opacity: 0, y: -20 }}
                                style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}
                            >
                                {/* Visual Preview Cards */}
                                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '16px' }}>
                                    <div className="glass card preview-card">
                                        <div className="preview-label"><Sparkles size={12} /> EFFICIENCY</div>
                                        <div className="preview-value">
                                            {((selectedRequest.tokens_saved / (selectedRequest.tokens_in + selectedRequest.tokens_saved)) * 100).toFixed(1)}%
                                        </div>
                                    </div>
                                    <div className="glass card preview-card">
                                        <div className="preview-label"><Clock size={12} /> LATENCY</div>
                                        <div className="preview-value">{selectedRequest.latency_ms.toFixed(0)}ms</div>
                                    </div>
                                    <div className="glass card preview-card">
                                        <div className="preview-label"><Layout size={12} /> RECONSTRUCTION</div>
                                        <div className="preview-value">{selectedRequest.reconstruction_log?.sequence?.length || 0} <span style={{ fontSize: '10px', opacity: 0.4 }}>chunks</span></div>
                                    </div>
                                </div>

                                {/* Reconstruction Context Preview (Few Cards) */}
                                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                                    <div style={{ fontSize: '11px', fontWeight: 700, opacity: 0.3, letterSpacing: '0.1em', marginBottom: '8px' }}>RECONSTRUCTION PREVIEW</div>
                                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                                        {selectedRequest.reconstruction_log?.sequence?.slice(0, 4).map((item, i) => (
                                            <div key={i} className="glass card" style={{ padding: '12px', border: '1px solid rgba(255,255,255,0.03)' }}>
                                                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
                                                    <span style={{ fontSize: '9px', fontWeight: 800, color: 'var(--accent)' }}>HIT {(item.score * 100).toFixed(0)}%</span>
                                                    <span style={{ fontSize: '9px', opacity: 0.2 }}>L-{item.line_index}</span>
                                                </div>
                                                <div style={{ fontSize: '12px', opacity: 0.6, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                                                    {item.text}
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </div>

                                {/* Navigation Button */}
                                <motion.button
                                    whileHover={{ scale: 1.02, backgroundColor: 'var(--fg)' }}
                                    whileTap={{ scale: 0.98 }}
                                    onClick={() => router.push(`/analytics/${selectedRequest.id}`)}
                                    className="go-to-btn"
                                >
                                    Explore Deep Observability <ArrowRight size={18} />
                                </motion.button>
                            </motion.div>
                        ) : !loading && (
                            <motion.div
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 0.2 }}
                                style={{ textAlign: 'center', padding: '100px 0', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '16px' }}
                            >
                                <Search size={40} />
                                <p style={{ fontSize: '14px' }}>No session selected</p>
                            </motion.div>
                        )}
                    </AnimatePresence>
                </div>

            </main>

            <style jsx>{`
        .custom-select-wrapper {
          position: relative;
          width: 100%;
        }
        .custom-select {
          width: 100%;
          appearance: none;
          padding: 16px 20px;
          border-radius: 14px;
          font-size: 14px;
          font-weight: 500;
          color: rgba(255,255,255,0.9);
          border: 1px solid rgba(255,255,255,0.05);
          cursor: pointer;
          transition: all 0.2s ease;
          outline: none;
        }
        .custom-select:hover {
          border-color: rgba(255,255,255,0.15);
          background: rgba(255,255,255,0.04);
        }
        .select-icon {
          position: absolute;
          right: 20px;
          top: 50%;
          transform: translateY(-50%);
          pointer-events: none;
          opacity: 0.3;
        }
        .preview-card {
           padding: 20px;
           text-align: center;
           display: flex;
           flex-direction: column;
           gap: 8px;
           background: rgba(255,255,255,0.01);
        }
        .preview-label {
           font-size: 10px;
           font-weight: 800;
           letter-spacing: 0.05em;
           opacity: 0.3;
           display: flex;
           align-items: center;
           justify-content: center;
           gap: 6px;
        }
        .preview-value {
           font-size: 24px;
           font-weight: 800;
           letter-spacing: -0.02em;
        }
        .go-to-btn {
           margin-top: 20px;
           width: 100%;
           height: 60px;
           background: #fff;
           color: #000;
           border: none;
           border-radius: 16px;
           font-size: 15px;
           font-weight: 700;
           cursor: pointer;
           display: flex;
           align-items: center;
           justify-content: center;
           gap: 12px;
           transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
        }
        option {
          background: #0a0a0a;
          color: #fff;
          padding: 10px;
        }
      `}</style>
        </div>
    );
}
