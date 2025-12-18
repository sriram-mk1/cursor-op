"use client";

import { useEffect } from "react";
import { createClient } from "@/utils/supabase";
import { useRouter } from "next/navigation";

export default function AuthCallback() {
    const router = useRouter();
    const supabase = createClient();

    useEffect(() => {
        const handleAuth = async () => {
            const { error } = await supabase.auth.getSession();
            if (!error) {
                router.push("/");
            } else {
                router.push("/login");
            }
        };
        handleAuth();
    }, [router, supabase.auth]);

    return (
        <div style={{ minHeight: "100vh", display: "flex", alignItems: "center", justifyContent: "center", background: "var(--bg)" }}>
            <div className="text-muted" style={{ fontSize: "14px" }}>Authenticating...</div>
        </div>
    );
}
