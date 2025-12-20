"use client";

import { useEffect, useState, useRef } from "react";
import { motion, AnimatePresence, useAnimation } from "framer-motion";
import {
    BarChart3, Layers, Database, Clock, Zap, MessageSquare,
    ArrowRight, Search, Maximize2, X, ChevronRight, Play, Archive
} from "lucide-react";
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

// --- Sub-components ---

function LogModal({ log, onClose }: { log: AnalyticsRow; onClose: () => void }) {
    if (!log) return null;

    const renderContent = (content: any) => {
        if (typeof content === 'string') return content;
        if (Array.isArray(content)) {
            return content.map((c, i) => (
                <span key={i}>{typeof c === 'string' ? c : (c.text || JSON.stringify(c))}</span>
            ));
        }
        return String(content || "");
    };

    return (
        <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="modal-backdrop"
            onClick={onClose}
        >
            <motion.div
                initial={{ scale: 0.95, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                exit={{ scale: 0.95, opacity: 0 }}
                className="modal-content glass"
                onClick={e => e.stopPropagation()}
                style={{ width: '900px', maxHeight: '80vh', display: 'flex', flexDirection: 'column' }}
            >
                <div style={{ padding: '20px', borderBottom: '1px solid rgba(255,255,255,0.05)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div>
                        <h3 style={{ fontSize: '15px', fontWeight: 600 }}>Raw Interaction Log</h3>
                        <p style={{ fontSize: '12px', opacity: 0.5 }}>{log.id}</p>
                    </div>
                    <button onClick={onClose} className="btn-icon"><X size={18} /></button>
                </div>

                <div style={{ padding: '24px', overflowY: 'auto', flex: 1, display: 'flex', flexDirection: 'column', gap: '24px' }}>
                    {log.raw_messages?.map((msg, i) => (
                        <div key={i} style={{ borderLeft: `2px solid ${msg.role === 'user' ? 'var(--accent)' : 'var(--purple)'}`, paddingLeft: '16px' }}>
                            <div style={{ fontSize: '10px', fontWeight: 800, textTransform: 'uppercase', marginBottom: '8px', opacity: 0.4 }}>{msg.role}</div>
                            <div style={{ fontSize: '13px', lineHeight: 1.6, opacity: 0.8, whiteSpace: 'pre-wrap' }}>
                                {renderContent(msg.content)}
                            </div>
                        </div>
                    ))}
                    {log.response_message && (
                        <div style={{ borderLeft: `2px solid var(--purple)`, paddingLeft: '16px' }}>
                            <div style={{ fontSize: '10px', fontWeight: 800, textTransform: 'uppercase', marginBottom: '8px', color: 'var(--purple)', opacity: 0.6 }}>ASSISTANT (FINAL)</div>
                            <div style={{ fontSize: '13px', lineHeight: 1.6, color: 'var(--fg)', whiteSpace: 'pre-wrap' }}>
                                {renderContent(log.response_message.content)}
                            </div>
                        </div>
                    )}
                </div>
            </motion.div>
        </motion.div>
    );
}

function ReconstructionAnimator({ sequence }: { sequence: any[] }) {
    const [isPlaying, setIsPlaying] = useState(false);
    const [visibleCount, setVisibleCount] = useState(0);

    useEffect(() => {
        let interval: any;
        if (isPlaying) {
            setVisibleCount(0);
            interval = setInterval(() => {
                setVisibleCount(prev => {
                    if (prev >= sequence.length) {
                        setIsPlaying(false);
                        return prev;
                    }
                    return prev + 1;
                });
            }, 100);
        }
        return () => clearInterval(interval);
    }, [isPlaying, sequence.length]);

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <h3 style={{ fontSize: '13px', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <Layers size={14} className="text-accent" /> Reconstruction Playback
                </h3>
                <button
                    onClick={() => setIsPlaying(true)}
                    disabled={isPlaying}
                    className="btn btn-primary"
                    style={{ height: '32px', fontSize: '11px', padding: '0 12px' }}
                >
                    <Play size={12} fill="currentColor" /> {isPlaying ? 'Playing...' : 'Replay Flow'}
                </button>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 40px 1fr', gap: '20px', minHeight: '300px' }}>
                {/* Source Column */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    <div style={{ fontSize: '10px', fontWeight: 700, opacity: 0.3, marginBottom: '4px' }}>ORIGINAL HISTORY (CHUNKS)</div>
                    {sequence?.map((item, i) => (
                        <motion.div
                            key={`src-${i}`}
                            initial={{ opacity: 0.3 }}
                            animate={{ opacity: isPlaying && i >= visibleCount ? 0.3 : 0.8 }}
                            className="glass card"
                            style={{ padding: '8px 12px', fontSize: '11px', border: '1px solid rgba(255,255,255,0.05)' }}
                        >
                            <div style={{ opacity: 0.3, fontSize: '9px', marginBottom: '2px' }}>LINE {item.line_index}</div>
                            <div style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{item.text}</div>
                        </motion.div>
                    ))}
                </div>

                <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center' }}>
                    <ArrowRight size={20} style={{ opacity: 0.2 }} />
                </div>

                {/* Target Column */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    <div style={{ fontSize: '10px', fontWeight: 700, opacity: 0.3, marginBottom: '4px' }}>RECONSTRUCTED CONTEXT</div>
                    <AnimatePresence>
                        {sequence?.slice(0, visibleCount).map((item, i) => (
                            <motion.div
                                key={`dst-${i}`}
                                initial={{ x: -100, scale: 0.8, opacity: 0 }}
                                animate={{ x: 0, scale: 1, opacity: 1 }}
                                className="glass card"
                                style={{ padding: '8px 12px', fontSize: '11px', border: '1px solid var(--accent-muted)', background: 'rgba(0,255,136,0.02)' }}
                            >
                                <div style={{ color: 'var(--accent)', fontSize: '9px', marginBottom: '2px', fontWeight: 700 }}>RELEVANCE: {(item.score * 100).toFixed(0)}%</div>
                                <div style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{item.text}</div>
                            </motion.div>
                        ))}
                    </AnimatePresence>
                </div>
            </div>
        </div>
    );
}

export default function ObservabilityPage() {
    const [conversations, setConversations] = useState<Conversation[]>([]);
    const [selectedConvo, setSelectedConvo] = useState<string | null>(null);
    const [requests, setRequests] = useState<AnalyticsRow[]>([]);
    const [selectedRequest, setSelectedRequest] = useState<AnalyticsRow | null>(null);
    const [loading, setLoading] = useState(true);

    const GATEWAY_URL = process.env.NEXT_PUBLIC_GATEWAY_URL || "http://localhost:8000";
    const supabase = createClient();

    useEffect(() => {
        const fetchStats = async () => {
            const { data: { user } } = await supabase.auth.getUser();
            if (user) {
                const resp = await fetch(`${GATEWAY_URL}/api/user/stats`, {
                    headers: { "x-user-id": user.id }
                });
                const data = await resp.json();
                setConversations(data.recent_conversations || []);
                setLoading(false);
            }
        };
        fetchStats();
    }, [supabase.auth]);

    const selectConversation = async (id: string) => {
        setSelectedConvo(id);
        setSelectedRequest(null);
        const resp = await fetch(`${GATEWAY_URL}/api/conversations/${id}`);
        const data = await resp.json();
        setRequests(data.requests || []);
    };

    return (
        <div className="app-container">
            <Sidebar />
            <main className="main-content" style={{ padding: '0px' }}>

                {/* Hero Section */}
                <section style={{ position: 'relative', background: 'var(--bg)', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                    <HeroSphere />
                    <div style={{
                        position: 'absolute',
                        top: '50%',
                        left: '50%',
                        transform: 'translate(-50%, -50%)',
                        textAlign: 'center',
                        pointerEvents: 'none'
                    }}>
                        <h1 style={{ fontSize: '32px', fontWeight: 700, letterSpacing: '-0.04em', marginBottom: '8px' }}>Deep Observability</h1>
                        <p style={{ fontSize: '14px', opacity: 0.4 }}>Real-time reconstruction analysis & flow mapping</p>
                    </div>
                </section>

                <div style={{ display: 'grid', gridTemplateColumns: '300px 1fr', minHeight: 'calc(100vh - 301px)' }}>

                    {/* Conversation Sidebar */}
                    <div style={{ borderRight: '1px solid rgba(255,255,255,0.05)', padding: '24px', background: 'rgba(255,255,255,0.01)' }}>
                        <h3 style={{ fontSize: '11px', fontWeight: 700, textTransform: 'uppercase', opacity: 0.3, marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <Archive size={12} /> Recent Conversations
                        </h3>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                            {conversations.map(convo => (
                                <button
                                    key={convo.id}
                                    onClick={() => selectConversation(convo.id)}
                                    className={`btn-list ${selectedConvo === convo.id ? 'active' : ''}`}
                                    style={{
                                        textAlign: 'left',
                                        padding: '12px',
                                        borderRadius: '8px',
                                        fontSize: '13px',
                                        background: selectedConvo === convo.id ? 'var(--accent-muted)' : 'transparent',
                                        color: selectedConvo === convo.id ? 'var(--accent)' : 'var(--fg)',
                                        display: 'flex',
                                        flexDirection: 'column',
                                        gap: '4px'
                                    }}
                                >
                                    <div style={{ fontWeight: 600, display: 'flex', alignItems: 'center', gap: '8px' }}>
                                        <MessageSquare size={14} /> {convo.title}
                                    </div>
                                    <div style={{ fontSize: '10px', opacity: 0.3 }}>
                                        <Clock size={10} style={{ display: 'inline', marginRight: '4px' }} />
                                        {new Date(convo.last_request_at * 1000).toLocaleString()}
                                    </div>
                                </button>
                            ))}
                        </div>
                    </div>

                    {/* Main Content Area */}
                    <div style={{ padding: '40px' }}>
                        <AnimatePresence mode="wait">
                            {!selectedConvo ? (
                                <motion.div
                                    initial={{ opacity: 0, y: 20 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    style={{ height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', opacity: 0.3 }}
                                >
                                    <Search size={48} style={{ marginBottom: '16px' }} />
                                    <p>Select a conversation to begin deep analysis</p>
                                </motion.div>
                            ) : (
                                <motion.div
                                    key={selectedConvo}
                                    initial={{ opacity: 0 }}
                                    animate={{ opacity: 1 }}
                                    style={{ display: 'flex', flexDirection: 'column', gap: '32px' }}
                                >
                                    {/* Request Selection (Horizontal Scroll or List) */}
                                    <section>
                                        <h3 style={{ fontSize: '12px', fontWeight: 700, marginBottom: '12px', opacity: 0.3 }}>INTERACTION TIMELINE</h3>
                                        <div style={{ display: 'flex', gap: '12px', overflowX: 'auto', paddingBottom: '12px' }}>
                                            {requests.map((req, idx) => (
                                                <button
                                                    key={req.id}
                                                    onClick={() => setSelectedRequest(req)}
                                                    className="glass card"
                                                    style={{
                                                        minWidth: '200px',
                                                        padding: '16px',
                                                        textAlign: 'left',
                                                        borderColor: selectedRequest?.id === req.id ? 'var(--accent)' : 'rgba(255,255,255,0.05)',
                                                        background: selectedRequest?.id === req.id ? 'rgba(0,255,136,0.02)' : 'rgba(255,255,255,0.02)'
                                                    }}
                                                >
                                                    <div style={{ fontSize: '10px', opacity: 0.4, marginBottom: '8px' }}>#{idx + 1} • {req.model}</div>
                                                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                                        <div style={{ fontSize: '14px', fontWeight: 700 }}>{req.tokens_saved} <span style={{ fontSize: '10px', opacity: 0.4 }}>saved</span></div>
                                                        <ArrowRight size={14} style={{ opacity: 0.3 }} />
                                                    </div>
                                                </button>
                                            ))}
                                        </div>
                                    </section>

                                    {/* Detail Section */}
                                    <AnimatePresence mode="wait">
                                        {selectedRequest && (
                                            <motion.div
                                                key={selectedRequest.id}
                                                initial={{ opacity: 0, y: 20 }}
                                                animate={{ opacity: 1, y: 0 }}
                                                className="detail-grid"
                                                style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '32px' }}
                                            >
                                                {/* Column 1: Metadata & Animation */}
                                                <div style={{ display: 'flex', flexDirection: 'column', gap: '32px' }}>
                                                    <section className="glass card" style={{ padding: '24px' }}>
                                                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '24px' }}>
                                                            <h2 style={{ fontSize: '18px', fontWeight: 800 }}>Request Details</h2>
                                                            <button
                                                                onClick={() => setSelectedRequest(selectedRequest)} // Open Modal logic
                                                                className="btn btn-outline"
                                                                style={{ height: '32px', padding: '0 12px', fontSize: '11px' }}
                                                            >
                                                                <Maximize2 size={12} /> Expand Raw
                                                            </button>
                                                        </div>

                                                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
                                                            <div style={{ padding: '16px', background: 'rgba(255,255,255,0.02)', borderRadius: '12px' }}>
                                                                <div style={{ fontSize: '10px', opacity: 0.4, marginBottom: '4px' }}>LATENCY</div>
                                                                <div style={{ fontSize: '16px', fontWeight: 700 }}>{selectedRequest.latency_ms.toFixed(0)}ms</div>
                                                            </div>
                                                            <div style={{ padding: '16px', background: 'rgba(255,255,255,0.02)', borderRadius: '12px' }}>
                                                                <div style={{ fontSize: '10px', opacity: 0.4, marginBottom: '4px' }}>OPTIMIZATION</div>
                                                                <div style={{ fontSize: '16px', fontWeight: 700, color: 'var(--accent)' }}>
                                                                    {((selectedRequest.tokens_saved / (selectedRequest.tokens_in + selectedRequest.tokens_saved)) * 100).toFixed(1)}%
                                                                </div>
                                                            </div>
                                                        </div>
                                                    </section>

                                                    <ReconstructionAnimator sequence={selectedRequest.reconstruction_log?.sequence || []} />
                                                </div>

                                                {/* Column 2: Summarized Interaction */}
                                                <section className="glass card" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '20px' }}>
                                                    <h2 style={{ fontSize: '13px', fontWeight: 700, opacity: 0.4, textTransform: 'uppercase' }}>Preview Flow</h2>
                                                    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '24px', overflowY: 'auto', maxHeight: '500px' }}>
                                                        {selectedRequest.raw_messages?.slice(-1).map((msg, i) => (
                                                            <div key={i} style={{ opacity: 0.7 }}>
                                                                <div style={{ fontSize: '10px', fontWeight: 700, marginBottom: '6px' }}>USER PROMPT</div>
                                                                <div style={{ fontSize: '13px', background: 'rgba(255,255,255,0.02)', padding: '16px', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.05)' }}>
                                                                    {msg.content?.toString().slice(0, 500)}...
                                                                </div>
                                                            </div>
                                                        ))}
                                                        {selectedRequest.response_message && (
                                                            <div>
                                                                <div style={{ fontSize: '10px', fontWeight: 700, marginBottom: '6px', color: 'var(--purple)' }}>AI RESPONSE</div>
                                                                <div style={{ fontSize: '13px', background: 'rgba(147, 51, 234, 0.05)', padding: '16px', borderRadius: '12px', border: '1px solid rgba(147, 51, 234, 0.2)' }}>
                                                                    {selectedRequest.response_message.content?.toString().slice(0, 500)}...
                                                                </div>
                                                            </div>
                                                        )}
                                                    </div>
                                                </section>
                                            </motion.div>
                                        )}
                                    </AnimatePresence>
                                </motion.div>
                            )}
                        </AnimatePresence>
                    </div>

                </div>

                {/* Modal Overlay */}
                <AnimatePresence>
                    {/* Add logic to show modal if needed */}
                </AnimatePresence>

            </main>

            <style jsx global>{`
        .modal-backdrop {
          position: fixed;
          inset: 0;
          background: rgba(0,0,0,0.8);
          backdrop-filter: blur(8px);
          z-index: 1000;
          display: flex;
          align-items: center;
          justify-content: center;
        }
        .modal-content {
          border: 1px solid rgba(255,255,255,0.1);
          box-shadow: 0 32px 64px -12px rgba(0,0,0,0.5);
          border-radius: 20px;
          overflow: hidden;
        }
        .btn-list {
          width: 100%;
          border: none;
          cursor: pointer;
          transition: all 0.2s;
        }
        .btn-list:hover {
          background: rgba(255,255,255,0.03);
        }
        .btn-list.active {
          background: var(--accent-muted);
        }
      `}</style>
        </div>
    );
}
