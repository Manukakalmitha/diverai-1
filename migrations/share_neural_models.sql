-- Enable Community Brain (Shared Models)
-- Allow any authenticated user to READ models, but only owners can WRITE (update/delete).

DROP POLICY IF EXISTS "Users can manage their own models" ON neural_models;

CREATE POLICY "Authenticated users can read all models" ON neural_models
    FOR SELECT USING (auth.role() = 'authenticated');

CREATE POLICY "Users can insert their own models" ON neural_models
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own models" ON neural_models
    FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own models" ON neural_models
    FOR DELETE USING (auth.uid() = user_id);
