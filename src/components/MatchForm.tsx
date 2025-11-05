import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Calculator, Loader2 } from "lucide-react";

export interface MatchData {
  battingTeam: string;
  bowlingTeam: string;
  venue: string;
  runsLeft: number;
  ballsLeft: number;
  wicketsLeft: number;
  totalRuns: number;
  currentRunRate: number;
  requiredRunRate: number;
  tossWinner: string;
  tossDecision: string;
}

interface MatchFormProps {
  onSubmit: (data: MatchData) => void;
  isLoading: boolean;
}

const teams = [
  "Mumbai Indians",
  "Chennai Super Kings",
  "Royal Challengers Bangalore",
  "Kolkata Knight Riders",
  "Delhi Capitals",
  "Punjab Kings",
  "Rajasthan Royals",
  "Sunrisers Hyderabad",
];

const venues = [
  "Wankhede Stadium, Mumbai",
  "M. Chinnaswamy Stadium, Bangalore",
  "Eden Gardens, Kolkata",
  "Feroz Shah Kotla, Delhi",
  "MA Chidambaram Stadium, Chennai",
];

export const MatchForm = ({ onSubmit, isLoading }: MatchFormProps) => {
  const [formData, setFormData] = useState<MatchData>({
    battingTeam: "",
    bowlingTeam: "",
    venue: "",
    runsLeft: 150,
    ballsLeft: 60,
    wicketsLeft: 7,
    totalRuns: 180,
    currentRunRate: 7.5,
    requiredRunRate: 9.0,
    tossWinner: "",
    tossDecision: "",
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(formData);
  };

  const updateField = (field: keyof MatchData, value: string | number) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  return (
    <Card className="shadow-[var(--shadow-card)]">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Calculator className="w-5 h-5 text-primary" />
          Match Details
        </CardTitle>
        <CardDescription>
          Enter current match parameters to predict win probability
        </CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Team Selection */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="battingTeam">Batting Team</Label>
              <Select value={formData.battingTeam} onValueChange={(v) => updateField("battingTeam", v)}>
                <SelectTrigger id="battingTeam">
                  <SelectValue placeholder="Select batting team" />
                </SelectTrigger>
                <SelectContent>
                  {teams.map(team => (
                    <SelectItem key={team} value={team}>{team}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label htmlFor="bowlingTeam">Bowling Team</Label>
              <Select value={formData.bowlingTeam} onValueChange={(v) => updateField("bowlingTeam", v)}>
                <SelectTrigger id="bowlingTeam">
                  <SelectValue placeholder="Select bowling team" />
                </SelectTrigger>
                <SelectContent>
                  {teams.map(team => (
                    <SelectItem key={team} value={team}>{team}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>

          {/* Venue */}
          <div className="space-y-2">
            <Label htmlFor="venue">Venue</Label>
            <Select value={formData.venue} onValueChange={(v) => updateField("venue", v)}>
              <SelectTrigger id="venue">
                <SelectValue placeholder="Select venue" />
              </SelectTrigger>
              <SelectContent>
                {venues.map(venue => (
                  <SelectItem key={venue} value={venue}>{venue}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* Match Statistics */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="space-y-2">
              <Label htmlFor="runsLeft">Runs Required</Label>
              <Input
                id="runsLeft"
                type="number"
                value={formData.runsLeft}
                onChange={(e) => updateField("runsLeft", parseInt(e.target.value))}
                min={1}
                max={300}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="ballsLeft">Balls Remaining</Label>
              <Input
                id="ballsLeft"
                type="number"
                value={formData.ballsLeft}
                onChange={(e) => updateField("ballsLeft", parseInt(e.target.value))}
                min={1}
                max={120}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="wicketsLeft">Wickets in Hand</Label>
              <Input
                id="wicketsLeft"
                type="number"
                value={formData.wicketsLeft}
                onChange={(e) => updateField("wicketsLeft", parseInt(e.target.value))}
                min={1}
                max={10}
              />
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="space-y-2">
              <Label htmlFor="totalRuns">Target Score</Label>
              <Input
                id="totalRuns"
                type="number"
                value={formData.totalRuns}
                onChange={(e) => updateField("totalRuns", parseInt(e.target.value))}
                min={1}
                max={300}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="currentRunRate">Current Run Rate</Label>
              <Input
                id="currentRunRate"
                type="number"
                step="0.1"
                value={formData.currentRunRate}
                onChange={(e) => updateField("currentRunRate", parseFloat(e.target.value))}
                min={0}
                max={20}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="requiredRunRate">Required Run Rate</Label>
              <Input
                id="requiredRunRate"
                type="number"
                step="0.1"
                value={formData.requiredRunRate}
                onChange={(e) => updateField("requiredRunRate", parseFloat(e.target.value))}
                min={0}
                max={20}
              />
            </div>
          </div>

          {/* Toss Information */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="tossWinner">Toss Winner</Label>
              <Select value={formData.tossWinner} onValueChange={(v) => updateField("tossWinner", v)}>
                <SelectTrigger id="tossWinner">
                  <SelectValue placeholder="Select toss winner" />
                </SelectTrigger>
                <SelectContent>
                  {teams.map(team => (
                    <SelectItem key={team} value={team}>{team}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label htmlFor="tossDecision">Toss Decision</Label>
              <Select value={formData.tossDecision} onValueChange={(v) => updateField("tossDecision", v)}>
                <SelectTrigger id="tossDecision">
                  <SelectValue placeholder="Select decision" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="bat">Bat First</SelectItem>
                  <SelectItem value="field">Field First</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          <Button 
            type="submit" 
            className="w-full" 
            size="lg" 
            disabled={isLoading || !formData.battingTeam || !formData.bowlingTeam}
            variant="hero"
          >
            {isLoading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Calculating...
              </>
            ) : (
              "Predict Win Probability"
            )}
          </Button>
        </form>
      </CardContent>
    </Card>
  );
};
