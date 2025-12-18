"use client";
import { useState, useEffect } from "react";
import { createClient } from "@/utils/supabase";
import { Sidebar } from "@/components/Sidebar";
import { Plus, Key, Copy, Trash2, ExternalLink, ShieldCheck, Eye, EyeOff } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
interface ApiKey {
    name: string;
    raw_key: string;
    hashed_key: string;
    created_at: number;
    total_tokens_saved: number;
    total_requests: number;
}
export default function KeysPage() {
    const [keys, setKeys] = useState<ApiKey[]>([]);
    const [isCreating, setIsCreating] = useState(false);
    const [newKeyName, setNewKeyName] = useState("");
    const [providerKey, setProviderKey] = useState("");
    const [user, setUser] = useState<any>(null);
    const [revealedKeys, setRevealedKeys] = useState<Set<string>>(new Set());
    const GATEWAY_URL = process.env.NEXT_PUBLIC_GATEWAY_URL || "http://localhost:8000";
    const supabase = createClient();
    useEffect(() => {
        const init = async () => {
            const { data: { user } } = await supabase.auth.getUser();
            setUser(user);
            if (user) fetchKeys(user.id);
        };
        init();
    }, []);
    const fetchKeys = async (userId: string) => {
        try {
            const resp = await fetch(`${GATEWAY_URL}/api/keys`, {
                headers: { "x-user-id": userId }
            });
            if (!resp.ok) {
                const errorData = await resp.json().catch(() => ({}));
                console.error(`API Error (${resp.status}):`, errorData);
                setKeys([]);
                return;
            }
            const data = await resp.json();
            if (Array.isArray(data)) {
                setKeys(data);
            } else {
                setKeys([]);
            }
        } catch (e) {
            console.error("Failed to fetch keys", e);
            setKeys([]);
        }
    };
    const toggleReveal = (hashedKey: string) => {
        const newRevealed = new Set(revealedKeys);
        if (newRevealed.has(hashedKey)) {
            newRevealed.delete(hashedKey);
        } else {
            newRevealed.add(hashedKey);
        }
        setRevealedKeys(newRevealed);
    };
    const handleCreateKey = async () => {
        if (!newKeyName || !user) return;
        try {
            const resp = await fetch(`${GATEWAY_URL}/api/keys`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ name: newKeyName, user_id: user.id }),
            });
            const newKeyData = await resp.json();
            alert(`New Key Created: ${newKeyData.key}\n\nYou can also reveal it anytime in the dashboard.`);
            setNewKeyName("");
            setIsCreating(false);
            fetchKeys(user.id);
        } catch (e) {
            console.error("Failed to create key", e);
        }
    };
    const handleDeleteKey = async (hashedKey: string) => {
        if (!confirm("Are you sure?") || !user) return;
        try {
            await fetch(`${GATEWAY_URL}/api/keys/${hashedKey}`, {
                method: "DELETE",
                headers: { "x-user-id": user.id }
            });
            fetchKeys(user.id);
        } catch (e) {
            console.error("Failed to delete key", e);
        }
    };
    const handleUpdateProvider = async () => {
        if (!user) return;
        const v1Key = localStorage.getItem("v1_key") || "v1-test-key";
        try {
            await fetch(`${GATEWAY_URL}/api/provider-key`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ v1_key: v1Key, user_id: user.id, openrouter_key: providerKey }),
            });
            alert("Provider key updated!");
            setProviderKey("");
        } catch (e) {
            console.error("Failed to update provider key", e);
        }
    };
    const copyToClipboard = (key: string) => {
        navigator.clipboard.writeText(key);
        localStorage.setItem("v1_key", key);
        alert("Key copied and set as active!");
    };
    return (
        <div className="layout-wrapper">
            <Sidebar />
            <main className="main-content">
                <header style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", marginBottom: "48px" }}>
                    <div>
                        <h1 style={{ fontSize: "24px", marginBottom: "4px" }}>API Keys</h1>
                        <p className="text-muted" style={{ fontSize: "14px" }}>Manage your V1 Gateway access keys and provider credentials.</p>
                    </div>
                    <button onClick={() => setIsCreating(true)} className="btn">
                        <Plus size={16} /> Create New Key
                    </button>
                </header>
                <div className="glass card" style={{ marginBottom: "32px", borderLeft: "4px solid var(--accent)" }}>
                    <div style={{ display: "flex", gap: "16px", alignItems: "center" }}>
                        <ShieldCheck className="text-accent" size={24} />
                        <div>
                            <div style={{ fontSize: "14px", fontWeight: 600, marginBottom: "4px" }}>Security Recommendation</div>
                            <div className="text-muted" style={{ fontSize: "12px" }}>Never share your API keys. Use different keys for different environments (dev, staging, prod).</div>
                        </div>
                    </div>
                </div>
                <div className="glass" style={{ overflow: "hidden" }}>
                    <div className="log-list">
                        <div className="log-item log-header" style={{ gridTemplateColumns: "1.5fr 2fr 1fr 1fr 0.5fr" }}>
                            <div>Name</div>
                            <div>API Key</div>
                            <div>Usage</div>
                            <div>Created</div>
                            <div style={{ textAlign: "right" }}>Actions</div>
                        </div>
                        {keys.map((key) => {
                            const isRevealed = revealedKeys.has(key.hashed_key);
                            const displayKey = isRevealed ? key.raw_key : "v1-••••••••••••••••";
                            return (
                                <div key={key.hashed_key} className="log-item" style={{ gridTemplateColumns: "1.5fr 2fr 1fr 1fr 0.5fr" }}>
                                    <div style={{ fontWeight: 500 }}>{key.name}</div>
                                    <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                                        <code style={{ background: "rgba(255,255,255,0.05)", padding: "2px 6px", borderRadius: "4px", fontSize: "11px", color: isRevealed ? "var(--accent)" : "rgba(255,255,255,0.4)" }}>
                                            {displayKey}
                                        </code>
                                        <button
                                            onClick={() => toggleReveal(key.hashed_key)}
                                            style={{ background: "none", border: "none", color: "rgba(255,255,255,0.3)", cursor: "pointer", padding: "4px" }}
                                            title={isRevealed ? "Hide Key" : "Reveal Key"}
                                        >
                                            {isRevealed ? <EyeOff size={14} /> : <Eye size={14} />}
                                        </button>
                                        <button
                                            onClick={() => copyToClipboard(key.raw_key)}
                                            style={{ background: "none", border: "none", color: "rgba(255,255,255,0.3)", cursor: "pointer", padding: "4px" }}
                                        >
                                            <Copy size={14} />
                                        </button>
                                    </div>
                                    <div className="text-muted">{key.total_requests} reqs</div>
                                    <div className="text-muted">{new Date(key.created_at * 1000).toLocaleDateString()}</div>
                                    <div style={{ textAlign: "right" }}>
                                        <button
                                            onClick={() => handleDeleteKey(key.hashed_key)}
                                            style={{ background: "none", border: "none", color: "#ff4444", cursor: "pointer", opacity: 0.5 }}
                                        >
                                            <Trash2 size={14} />
                                        </button>
                                    </div>
                                </div>
                            );
                        })}
                        {keys.length === 0 && (
                            <div style={{ padding: "40px", textAlign: "center", opacity: 0.3 }}>
                                No API keys found. Create one to get started.
                            </div>
                        )}
                    </div>
                </div>
                <div className="glass card" style={{ marginTop: "32px" }}>
                    <div className="card-header">
                        <h3 style={{ fontSize: "16px" }}>Provider Credentials</h3>
                    </div>
                    <p className="text-muted" style={{ fontSize: "13px", marginBottom: "20px" }}>
                        Link your OpenRouter API key to your V1 account to enable context optimization.
                    </p>
                    <div style={{ display: "flex", gap: "12px" }}>
                        <input
                            type="password"
                            className="input"
                            placeholder="sk-or-v1-..."
                            style={{ flex: 1 }}
                            value={providerKey}
                            onChange={(e) => setProviderKey(e.target.value)}
                        />
                        <button onClick={handleUpdateProvider} className="btn">Update Key</button>
                    </div>
                </div>
            </main>
            <AnimatePresence>
                {isCreating && (
                    <div className="modal-overlay" onClick={() => setIsCreating(false)}>
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: 20 }}
                            className="modal-content"
                            onClick={(e) => e.stopPropagation()}
                        >
                            <h2 style={{ fontSize: "20px", marginBottom: "8px" }}>Create API Key</h2>
                            <p className="text-muted" style={{ fontSize: "13px", marginBottom: "24px" }}>
                                Give your new key a name to identify it later.
                            </p>

                            <div style={{ display: "grid", gap: "20px" }}>
                                <div>
                                    <label className="stat-label" style={{ display: "block", marginBottom: "8px" }}>Key Name</label>
                                    <input
                                        type="text"
                                        className="input"
                                        placeholder="e.g. Production"
                                        style={{ width: "100%" }}
                                        value={newKeyName}
                                        onChange={(e) => setNewKeyName(e.target.value)}
                                        autoFocus
                                    />
                                </div>

                                <div style={{ display: "flex", gap: "12px", justifyContent: "flex-end", marginTop: "12px" }}>
                                    <button onClick={() => setIsCreating(false)} className="btn btn-outline" style={{ padding: "10px 20px" }}>Cancel</button>
                                    <button onClick={handleCreateKey} className="btn" style={{ padding: "10px 24px" }}>Create Key</button>
                                </div>
                            </div>
                        </motion.div>
                    </div>
                )}
            </AnimatePresence>
        </div>
    );
}
