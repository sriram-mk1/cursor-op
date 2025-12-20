"use client";

import { useEffect, useState, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
    ChevronDown, ArrowRight, Clock,
    Sparkles, Layout, Search, Zap, Code
} from "lucide-react";
import { useRouter } from "next/navigation";
import { Sidebar } from "@/components/Sidebar";
import { HeroCubes } from "@/components/HeroCubes";
import { createClient } from "@/utils/supabase";

// --- Types ---
interface AnalyticsRow {
    id: string;
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
    const [requests, setRequests] = useState<AnalyticsRow[]>([]);
    const [selectedRequest, setSelectedRequest] = useState<AnalyticsRow | null>(null);
    const [loading, setLoading] = useState(true);

    const GATEWAY_URL = process.env.NEXT_PUBLIC_GATEWAY_URL || "http://localhost:8000";
    const supabase = createClient();

    useEffect(() => {
        const fetchRequests = async () => {
            const { data: { user } } = await supabase.auth.getUser();
            if (user) {
                try {
                    const resp = await fetch(`${GATEWAY_URL}/api/user/stats`, {
                        headers: { "x-user-id": user.id }
                    });
                    const data = await resp.json();
                    const sorted = (data.recent_requests || []).sort((a: any, b: any) => b.timestamp - a.timestamp);
                    setRequests(sorted);
                    if (sorted.length > 0) setSelectedRequest(sorted[0]);
                } catch (e) {
                    console.error("Failed to fetch requests", e);
                }
                setLoading(false);
            }
        };
        fetchRequests();
    }, [supabase.auth]);

    return (
        <div className="app-container">
            <Sidebar />
            <main className="main-content" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', paddingTop: '40px', gap: '0px' }}>

                {/* Animated Cubes Above Title */}
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ duration: 1 }}
                    style={{ width: '100%', maxWidth: '500px', height: '240px' }}
                >
                    <HeroCubes />
                </motion.div>

                {/* Minimalist Title */}
                <div style={{ textAlign: 'center', marginBottom: '40px' }}>
                    <motion.h1
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        style={{
                            fontSize: '36px',
                            fontWeight: 800,
                            letterSpacing: '-0.05em',
                            marginBottom: '8px',
                            color: '#fff'
                        }}
                    >
                        Deep Observability
                    </motion.h1>
                    <motion.p
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 0.4 }}
                        style={{ fontSize: '14px', fontWeight: 500 }}
                    >
                        Real-time context reconstruction analysis
                    </motion.p>
                </div>

                {/* Request Selection Dropdown */}
                <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    style={{ width: '100%', maxWidth: '500px', marginBottom: '40px' }}
                >
                    <div className="custom-select-wrapper">
                        <select
                            value={selectedRequest?.id || ""}
                            onChange={(e) => setSelectedRequest(requests.find(r => r.id === e.target.value) || null)}
                            className="glass custom-select"
                        >
                            {requests.map((r, i) => (
                                <option key={r.id} value={r.id}>
                                    {new Date(r.timestamp * 1000).toLocaleString()} • {r.model}
                                </option>
                            ))}
                            {requests.length === 0 && <option value="" disabled>No requests found</option>}
                        </select>
                        <ChevronDown className="select-icon" size={16} />
                    </div>
                </motion.div>

                {/* Preview Section - Bento Box Style (like Analytics page) */}
                <div style={{ width: '100%', maxWidth: '1000px', padding: '0 20px' }}>
                    <AnimatePresence mode="wait">
                        {selectedRequest ? (
                            <motion.div
                                key={selectedRequest.id}
                                initial={{ opacity: 0, scale: 0.98 }}
                                animate={{ opacity: 1, scale: 1 }}
                                exit={{ opacity: 0, scale: 0.98 }}
                                style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}
                            >
                                {/* Metrics Row */}
                                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px' }}>
                                    <div className="glass bento-item">
                                        <div className="bento-label"><Zap size={12} /> SAVINGS</div>
                                        <div className="bento-value">
                                            {((selectedRequest.tokens_saved / (selectedRequest.tokens_in + selectedRequest.tokens_saved)) * 100).toFixed(1)}%
                                        </div>
                                    </div>
                                    <div className="glass bento-item">
                                        <div className="bento-label"><Clock size={12} /> LATENCY</div>
                                        <div className="bento-value">{selectedRequest.latency_ms.toFixed(0)}<span style={{ fontSize: '12px' }}>ms</span></div>
                                    </div>
                                    <div className="glass bento-item">
                                        <div className="bento-label"><Layout size={12} /> CHUNKS</div>
                                        <div className="bento-value">{selectedRequest.reconstruction_log?.sequence?.length || 0}</div>
                                    </div>
                                    <div className="glass bento-item">
                                        <div className="bento-label"><Code size={12} /> TOKENS</div>
                                        <div className="bento-value">{selectedRequest.tokens_in}</div>
                                    </div>
                                </div>

                                {/* Reconstruction Preview Chunks */}
                                <div className="glass" style={{ padding: '24px', borderRadius: '24px' }}>
                                    <div style={{ fontSize: '11px', fontWeight: 800, opacity: 0.3, letterSpacing: '0.1em', marginBottom: '20px' }}>CONTEXT RECONSTRUCTION SEQUENCE</div>
                                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                                        {selectedRequest.reconstruction_log?.sequence?.slice(0, 4).map((item, i) => (
                                            <div key={i} className="glass" style={{ padding: '16px', borderRadius: '16px', border: '1px solid rgba(255,255,255,0.03)', background: 'rgba(255,255,255,0.01)' }}>
                                                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                                                    <span style={{ fontSize: '10px', fontWeight: 800, color: i < 2 ? 'var(--accent)' : 'rgba(255,255,255,0.4)' }}>
                                                        {item.score ? `HIT ${(item.score * 100).toFixed(0)}%` : 'HISTORY'}
                                                    </span>
                                                    <span style={{ fontSize: '10px', opacity: 0.2 }}>L-{item.line_index}</span>
                                                </div>
                                                <div style={{ fontSize: '13px', opacity: 0.6, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', fontFamily: 'monospace' }}>
                                                    {item.text}
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </div>

                                {/* Big Go-To Button */}
                                <motion.button
                                    whileHover={{ scale: 1.01, backgroundColor: '#fff', color: '#000' }}
                                    whileTap={{ scale: 0.99 }}
                                    onClick={() => router.push(`/analytics/${selectedRequest.id}`)}
                                    style={{
                                        height: '70px',
                                        borderRadius: '24px',
                                        background: 'rgba(255,255,255,0.05)',
                                        border: '1px solid rgba(255,255,255,0.1)',
                                        color: '#fff',
                                        fontSize: '16px',
                                        fontWeight: 700,
                                        display: 'flex',
                                        alignItems: 'center',
                                        justifyContent: 'center',
                                        gap: '12px',
                                        cursor: 'pointer',
                                        transition: 'all 0.2s ease',
                                        marginTop: '10px'
                                    }}
                                >
                                    Enter Deep Observer <ArrowRight size={20} />
                                </motion.button>

                            </motion.div>
                        ) : !loading && (
                            <div style={{ textAlign: 'center', padding: '100px 0', opacity: 0.2 }}>
                                <Search size={40} style={{ margin: '0 auto 16px' }} />
                                <p>No requests found for this account</p>
                            </div>
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
          padding: 18px 24px;
          border-radius: 20px;
          font-size: 15px;
          font-weight: 600;
          color: #fff;
          border: 1px solid rgba(255,255,255,0.08);
          cursor: pointer;
          background: rgba(255,255,255,0.02);
          transition: all 0.2s ease;
          outline: none;
        }
        .custom-select:hover {
          background: rgba(255,255,255,0.05);
          border-color: rgba(255,255,255,0.2);
        }
        .select-icon {
          position: absolute;
          right: 24px;
          top: 50%;
          transform: translateY(-50%);
          pointer-events: none;
          opacity: 0.4;
        }
        .bento-item {
           padding: 24px;
           border-radius: 24px;
           text-align: left;
           background: rgba(255,255,255,0.02);
        }
        .bento-label {
           font-size: 10px;
           font-weight: 900;
           letter-spacing: 0.1em;
           opacity: 0.3;
           display: flex;
           align-items: center;
           gap: 8px;
           margin-bottom: 12px;
        }
        .bento-value {
           font-size: 28px;
           font-weight: 800;
           letter-spacing: -0.03em;
        }
        option {
          background: #000;
          color: #fff;
        }
      `}</style>
        </div>
    );
}
