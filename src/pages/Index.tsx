import { useState, useRef } from "react";
import { Hero } from "@/components/Hero";
import { MatchForm, MatchData } from "@/components/MatchForm";
import { PredictionResult } from "@/components/PredictionResult";
import { ShapExplanation } from "@/components/ShapExplanation";
import { toast } from "sonner";
import { api, PredictionResponse } from "@/lib/api";

const Index = () => {
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [matchData, setMatchData] = useState<MatchData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const formRef = useRef<HTMLDivElement>(null);

  const handleGetStarted = () => {
    formRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  const handleSubmit = async (data: MatchData) => {
    setIsLoading(true);
    setMatchData(data);
    
    try {
      // Map frontend data to API format
      const apiRequest = {
        team1: data.battingTeam,
        team2: data.bowlingTeam,
        venue: data.venue,
        toss_winner: data.tossWinner || data.battingTeam,
        toss_decision: data.tossDecision || 'bat',
        match_type: 'T20', // Default to T20, can be made dynamic
        runs_required: data.runsLeft,
        balls_remaining: data.ballsLeft,
        wickets_in_hand: data.wicketsLeft,
        target_match: data.totalRuns,
        current_run_rate: data.currentRunRate,
        required_run_rate: data.requiredRunRate,
      };

    // Call the FastAPI backend
    const result = await api.predict(apiRequest);
    setPrediction(result);
      toast.success("Prediction calculated successfully!");
      
      // Scroll to results
      setTimeout(() => {
        window.scrollTo({
          top: document.documentElement.scrollHeight,
          behavior: 'smooth'
        });
      }, 100);
      
    } catch (error) {
      console.error('Prediction error:', error);
      toast.error(error instanceof Error ? error.message : "Failed to calculate prediction. Please check if the backend is running.");
    } finally {
      setIsLoading(false);
    }
  };

  // Convert SHAP values from API to component format
  const shapFeatures = prediction?.shap_explanation.map(shap => ({
    feature: shap.feature,
    impact: Math.abs(shap.value),
    direction: shap.impact as "positive" | "negative",
  })) || [];

  return (
    <div className="min-h-screen bg-background">
      <Hero onGetStarted={handleGetStarted} />
      
      <main className="container mx-auto px-4 py-16 space-y-12">
        <div ref={formRef} className="max-w-4xl mx-auto">
          <MatchForm onSubmit={handleSubmit} isLoading={isLoading} />
        </div>

        {prediction && matchData && (
          <div className="max-w-4xl mx-auto space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
            <PredictionResult
              battingTeam={matchData.battingTeam}
              bowlingTeam={matchData.bowlingTeam}
              winProbability={(() => {
                const raw = Number(prediction.probability ?? 0);
                // Defensive normalization:
                // - If backend returns [0,1] (e.g., 0.73), multiply by 100
                // - If backend returns [1,100], use as-is
                // - If backend returns >100 (unexpected), divide by 100
                let normalized = raw;
                if (raw <= 1) normalized = raw * 100;
                else if (raw > 100) normalized = raw / 100;
                // Cap between 0 and 100
                normalized = Math.max(0, Math.min(100, normalized));
                const rounded = Math.round(normalized * 100) / 100; // keep max two decimals
                return rounded;
              })()}
            />
            
            <ShapExplanation features={shapFeatures} />
            
            {/* Additional Information Card */}
            <div className="bg-muted/50 rounded-lg p-6 space-y-4">
              <h3 className="text-lg font-semibold">Prediction Details</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div>
                  <div className="text-muted-foreground">Winner</div>
                  <div className="font-semibold text-lg">{prediction.winner}</div>
                </div>
                <div>
                  <div className="text-muted-foreground">Confidence</div>
                  <div className="font-semibold text-lg capitalize">{prediction.confidence}</div>
                </div>
                <div>
                  <div className="text-muted-foreground">Toss</div>
                  <div className="font-semibold">{prediction.factors.toss}</div>
                </div>
                <div>
                  <div className="text-muted-foreground">Venue</div>
                  <div className="font-semibold">{prediction.factors.venue}</div>
                </div>
              </div>
            </div>
          </div>
        )}

        {!prediction && (
          <div className="max-w-4xl mx-auto text-center py-12 text-muted-foreground">
            <p>Enter match details above to see AI-powered predictions and explainable insights.</p>
          </div>
        )}
      </main>

      <footer className="border-t mt-20">
        <div className="container mx-auto px-4 py-8">
          <div className="text-center text-sm text-muted-foreground">
            <p>© 2025 Cricket Win Predictor. Powered by Machine Learning & Explainable AI.</p>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Index;
