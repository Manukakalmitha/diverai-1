import React, { createContext, useContext, useState, useEffect } from 'react';
import { supabase } from '../lib/supabase';

const AppContext = createContext();

export const AppProvider = ({ children }) => {
    const [user, setUser] = useState(null);
    const [profile, setProfile] = useState(null);
    const [neuralState, setNeuralState] = useState({ alpha: 0.35, beta: 0.30, gamma: 0.20, delta: 0.10, omega: 0.50, iterations: 0 });
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        // Fetch global neural state
        fetchNeuralState();
        // Check initial session
        supabase.auth.getSession().then(({ data: { session } }) => {
            setUser(session?.user || null);
            if (session?.user) fetchProfile(session.user.id, session.user.email);
            else setLoading(false);
        });

        // Listen for auth changes
        const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, session) => {
            setUser(session?.user || null);
            if (session?.user) {
                fetchProfile(session.user.id, session.user.email);
            } else {
                setProfile(null);
                setLoading(false);
            }
        });

        return () => subscription.unsubscribe();
    }, []);

    const fetchProfile = async (userId, userEmail) => {
        try {
            let { data, error } = await supabase.from('profiles').select('*').eq('id', userId).single();

            if (error && error.code === 'PGRST116') {
                // Create profile if missing
                const { data: newProfile, error: insertError } = await supabase
                    .from('profiles')
                    .insert([{
                        id: userId,
                        email: userEmail,
                        subscription_tier: 'free',
                        upload_count: 0,
                        last_upload_date: new Date().toISOString().split('T')[0]
                    }])
                    .select()
                    .single();

                if (!insertError) data = newProfile;
            }
            setProfile(data);
        } catch (err) {
            console.error("Profile fetch error:", err);
        } finally {
            setLoading(false);
        }
    };

    const refreshProfile = () => {
        if (user) fetchProfile(user.id, user.email);
    };

    const fetchNeuralState = async () => {
        try {
            const { data, error } = await supabase
                .from('neural_state')
                .select('*')
                .eq('id', 'global_master')
                .single();

            if (data && !error) {
                setNeuralState(data);
            }
        } catch (err) {
            console.error("Neural state sync error:", err);
        }
    };

    return (
        <AppContext.Provider value={{
            user, profile, neuralState, loading,
            refreshProfile, fetchNeuralState, setUser, setProfile, setNeuralState
        }}>
            {children}
        </AppContext.Provider>
    );
};

export const useAppContext = () => useContext(AppContext);
