"use client";

import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
    Search, Zap, MessageSquare, ChevronRight, Plus, Hash, Clock
} from "lucide-react";
import { useRouter } from "next/navigation";
import { Sidebar } from "@/components/Sidebar";
import { createClient } from "@/utils/supabase";

interface Conversation {
    id: string;
    session_id: string;
    title: string;
    last_request_at: number;
    metadata: any;
}

export default function ObservabilityListPage() {
    const router = useRouter();
    const [conversations, setConversations] = useState<Conversation[]>([]);
    const [loading, setLoading] = useState(true);

    const GATEWAY_URL = process.env.NEXT_PUBLIC_GATEWAY_URL || "http://localhost:8000";
    const supabase = createClient();

    useEffect(() => {
        const fetchConversations = async () => {
            const { data: { user } } = await supabase.auth.getUser();
            const userId = user?.id || (process.env.NODE_ENV === 'development' ? 'dev-user' : null);

            if (userId) {
                try {
                    const resp = await fetch(`${GATEWAY_URL}/api/user/conversations`, {
                        headers: { "x-user-id": userId }
                    });
                    const data = await resp.json();

                    if (data.length === 0 && process.env.NODE_ENV === 'development') {
                        setConversations([
                            { id: 'c1', session_id: 'sess-1', title: 'How to implement RAG with role preservation?', last_request_at: Date.now() / 1000, metadata: {} },
                            { id: 'c2', session_id: 'sess-2', title: 'Fixing Modal TypeError in FastAPI', last_request_at: Date.now() / 1000 - 3600, metadata: {} },
                            { id: 'c3', session_id: 'sess-3', title: 'Why are my Supabase logs disappearing?', last_request_at: Date.now() / 1000 - 7200, metadata: {} }
                        ]);
                    } else {
                        setConversations(data);
                    }
                } catch (e) {
                    console.error("Failed to fetch conversations", e);
                }
                setLoading(false);
            }
        };
        fetchConversations();
    }, [GATEWAY_URL, supabase.auth]);

    return (
        <div className="layout-wrapper">
            <Sidebar />
            <main className="main-content">

                <header style={{ marginBottom: "48px" }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '8px' }}>
                        <Hash size={20} className="text-accent" />
                        <h1 style={{ fontSize: "24px", fontWeight: 700, letterSpacing: '-0.03em', margin: 0 }}>Active Threads</h1>
                    </div>
                    <p className="text-muted" style={{ fontSize: "14px" }}>
                        Conversation threads isolated by your unique fingerprint.
                    </p>
                </header>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(400px, 1fr))', gap: '20px', maxWidth: '1200px' }}>
                    <AnimatePresence>
                        {conversations.map((convo, idx) => (
                            <motion.div
                                key={convo.id}
                                initial={{ opacity: 0, y: 10 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ delay: idx * 0.05 }}
                                onClick={() => router.push(`/analytics/${convo.session_id}`)}
                                className="glass card"
                                style={{
                                    padding: '24px',
                                    cursor: 'pointer',
                                    border: '1px solid rgba(255,255,255,0.05)',
                                    display: 'flex',
                                    flexDirection: 'column',
                                    gap: '16px',
                                    position: 'relative',
                                    overflow: 'hidden'
                                }}
                                whileHover={{
                                    scale: 1.01,
                                    borderColor: 'var(--accent)',
                                    background: 'rgba(255,255,255,0.03)'
                                }}
                            >
                                {/* Isolation Badge */}
                                <div style={{
                                    position: 'absolute',
                                    top: '0',
                                    right: '0',
                                    padding: '4px 8px',
                                    background: 'var(--accent)',
                                    color: 'black',
                                    fontSize: '9px',
                                    fontWeight: 900,
                                    letterSpacing: '0.1em'
                                }}>
                                    ISOLATED
                                </div>

                                <div style={{ display: 'flex', gap: '12px', alignItems: 'flex-start' }}>
                                    <div className="glass" style={{ padding: '8px', borderRadius: '10px', background: 'rgba(var(--accent-rgb), 0.1)' }}>
                                        <MessageSquare size={16} className="text-accent" />
                                    </div>
                                    <div style={{ flex: 1 }}>
                                        <h3 style={{ fontSize: '15px', fontWeight: 600, lineHeight: 1.4, color: 'var(--fg)', marginBottom: '4px' }}>
                                            {convo.title || "Untitled Conversation"}
                                        </h3>
                                        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', fontSize: '11px', opacity: 0.4 }}>
                                            <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                                                <Clock size={10} />
                                                {new Date(convo.last_request_at * 1000).toLocaleDateString()}
                                            </div>
                                            <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                                                <Zap size={10} className="text-accent" />
                                                ID: {convo.session_id.slice(0, 8)}...
                                            </div>
                                        </div>
                                    </div>
                                    <ChevronRight size={18} style={{ opacity: 0.2 }} />
                                </div>

                                <div style={{
                                    marginTop: 'auto',
                                    paddingTop: '16px',
                                    borderTop: '1px solid rgba(255,255,255,0.05)',
                                    display: 'flex',
                                    justifyContent: 'space-between',
                                    alignItems: 'center'
                                }}>
                                    <span style={{ fontSize: '10px', fontWeight: 700, opacity: 0.3, letterSpacing: '0.05em' }}>
                                        VOYAGE OPTIMIZATION ACTIVE
                                    </span>
                                    <div className="text-accent" style={{ fontSize: '11px', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '4px' }}>
                                        View Pipeline <ChevronRight size={12} />
                                    </div>
                                </div>
                            </motion.div>
                        ))}
                    </AnimatePresence>

                    {!loading && conversations.length === 0 && (
                        <div style={{ gridColumn: '1/-1', textAlign: 'center', padding: '100px 0', opacity: 0.2 }}>
                            <Search size={48} style={{ margin: '0 auto 16px' }} />
                            <p>No active threads identified.</p>
                        </div>
                    )}
                </div>
            </main>
        </div>
    );
}
