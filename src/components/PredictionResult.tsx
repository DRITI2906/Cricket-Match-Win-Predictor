import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { TrendingUp, TrendingDown, Target } from "lucide-react";

interface PredictionResultProps {
  battingTeam: string;
  bowlingTeam: string;
  winProbability: number;
}

export const PredictionResult = ({ battingTeam, bowlingTeam, winProbability }: PredictionResultProps) => {
  const losingProbability = 100 - winProbability;
  const isHighConfidence = winProbability > 70 || winProbability < 30;

  return (
    <Card className="shadow-[var(--shadow-card)] border-2 border-primary/20">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-2xl">
          <Target className="w-6 h-6 text-primary" />
          Win Prediction
        </CardTitle>
        <CardDescription>AI-powered match outcome probability</CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Main Prediction */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="space-y-1">
              <div className="text-sm font-medium text-muted-foreground">Batting Team</div>
              <div className="text-xl font-bold">{battingTeam}</div>
            </div>
            <div className="flex items-center gap-2">
              {winProbability > 50 ? (
                <TrendingUp className="w-8 h-8 text-primary" />
              ) : (
                <TrendingDown className="w-8 h-8 text-destructive" />
              )}
              <div className="text-5xl font-bold bg-gradient-to-r from-primary to-primary/70 bg-clip-text text-transparent">
                {winProbability}%
              </div>
            </div>
          </div>

          <Progress value={winProbability} className="h-4" />

          <div className="flex items-center justify-between text-sm">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-primary" />
              <span className="font-medium">{battingTeam}</span>
              <span className="text-muted-foreground">{winProbability}%</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-muted-foreground">{losingProbability}%</span>
              <span className="font-medium">{bowlingTeam}</span>
              <div className="w-3 h-3 rounded-full bg-muted" />
            </div>
          </div>
        </div>

        {/* Confidence Badge */}
        <div className={`p-4 rounded-lg ${isHighConfidence ? 'bg-primary/10' : 'bg-muted'}`}>
          <div className="flex items-center justify-between">
            <div className="space-y-1">
              <div className="text-sm font-medium">Model Confidence</div>
              <div className="text-xs text-muted-foreground">
                {isHighConfidence 
                  ? "High confidence in prediction" 
                  : "Moderate confidence - match could swing either way"}
              </div>
            </div>
            <div className={`text-2xl font-bold ${isHighConfidence ? 'text-primary' : 'text-muted-foreground'}`}>
              {isHighConfidence ? "High" : "Medium"}
            </div>
          </div>
        </div>

        {/* Quick Stats */}
        <div className="grid grid-cols-3 gap-4 pt-4 border-t">
          <div className="text-center space-y-1">
            <div className="text-2xl font-bold text-primary">{winProbability}%</div>
            <div className="text-xs text-muted-foreground">Win Chance</div>
          </div>
          <div className="text-center space-y-1">
            <div className="text-2xl font-bold text-accent">
              {isHighConfidence ? "Strong" : "Balanced"}
            </div>
            <div className="text-xs text-muted-foreground">Position</div>
          </div>
          <div className="text-center space-y-1">
            <div className="text-2xl font-bold text-secondary">Live</div>
            <div className="text-xs text-muted-foreground">Status</div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};
