import React, { useState, useMemo, useCallback, useEffect } from 'react';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer,
  PieChart, Pie, Cell, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, Legend,
} from 'recharts';
import {
  LayoutDashboard, Compass, Briefcase, ShieldAlert, Settings, Zap, Loader2, Search,
  ArrowUpRight, ArrowDownRight, Minus, TrendingUp, TrendingDown, Wallet, Target, Activity,
  ChevronUp, ChevronDown, ArrowUpDown,
} from 'lucide-react';

const HURDLE_RATE = 0.10; 
const DONUT_COLORS = ['#38bdf8', '#34d399', '#a78bfa', '#fbbf24', '#fb7185', '#22d3ee', '#818cf8', '#4ade80', '#f472b6', '#facc15'];
const RANGES = ['1M', '3M', '6M', 'YTD', '1Y'];

const RISK_METRICS = [
  { metric: 'Alpha Capture', portfolio: 88, benchmark: 50 },
  { metric: 'Sortino (Downside)', portfolio: 76, benchmark: 45 },
  { metric: 'VaR (95%) Limit', portfolio: 82, benchmark: 60 },
  { metric: 'Calmar Ratio', portfolio: 71, benchmark: 40 },
  { metric: 'Beta Hedging', portfolio: 85, benchmark: 50 },
];

const inr = (num, decimals = 0) => new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR', maximumFractionDigits: decimals }).format(num);
const inrCompact = (num) => {
  if (Math.abs(num) >= 10000000) return `₹${(num / 10000000).toFixed(2)}Cr`;
  if (Math.abs(num) >= 100000) return `₹${(num / 100000).toFixed(1)}L`;
  return inr(num);
};

const getSignal = (predictedReturn) => {
  if (predictedReturn > HURDLE_RATE) return 'BUY';
  if (predictedReturn < -HURDLE_RATE) return 'SELL';
  return 'HOLD';
};

const SIGNAL_META = {
  BUY:  { classes: 'bg-emerald-400/10 text-emerald-400 border-emerald-400/30', Icon: ArrowUpRight },
  HOLD: { classes: 'bg-amber-400/10 text-amber-400 border-amber-400/30',       Icon: Minus },
  SELL: { classes: 'bg-rose-400/10 text-rose-400 border-rose-400/30',          Icon: ArrowDownRight },
};

function convictionColor(score) {
  if (score >= 75) return { bar: 'bg-emerald-400', text: 'text-emerald-400' };
  if (score >= 50) return { bar: 'bg-sky-400', text: 'text-sky-400' };
  return { bar: 'bg-rose-400/80', text: 'text-rose-400' };
}

function generateGrowthSeries() {
  const data = [];
  let value = 1000000;
  let seed = 1337;
  const rand = () => { seed = (seed * 9301 + 49297) % 233280; return seed / 233280; };
  const today = new Date(); 
  for (let i = 364; i >= 0; i--) {
    const d = new Date(today);
    d.setDate(d.getDate() - i);
    const drift = 0.00058;
    const shock = (rand() - 0.485) * 0.017;
    value = Math.max(value * (1 + drift + shock), 400000);
    data.push({
      label: d.toLocaleDateString('en-IN', { day: '2-digit', month: 'short' }),
      value: Math.round(value),
      time: d.getTime(),
      isYTD: d.getFullYear() === today.getFullYear(),
    });
  }
  return data;
}

function GlassCard({ children, className = '' }) {
  return <div className={`bg-slate-900/60 backdrop-blur-xl border border-slate-800 rounded-2xl shadow-lg ${className}`}>{children}</div>;
}

export default function App() {
  const [activeView, setActiveView] = useState('overview');
  const [isRunning, setIsRunning] = useState(false);
  const [lastSync, setLastSync] = useState('--:--:--');
  const [assetUniverse, setAssetUniverse] = useState([]);
  
  // FETCH DATA FROM FASTAPI BACKEND
  const fetchMarketData = async () => {
    setIsRunning(true);
    try {
      const response = await fetch('http://localhost:8000/api/market-opportunities');
      if (!response.ok) throw new Error("API Error");
      const data = await response.json();
      setAssetUniverse(data);
      setLastSync(new Date().toLocaleTimeString('en-IN', { hour12: false }));
    } catch (error) {
      console.error("Failed to fetch data from Python backend:", error);
    } finally {
      setIsRunning(false);
    }
  };

  useEffect(() => {
    fetchMarketData();
  }, []);

  const fullSeries = useMemo(() => generateGrowthSeries(), []);
  const [range, setRange] = useState('6M');

  const visibleSeries = useMemo(() => {
    if (range === '1M') return fullSeries.slice(-30);
    if (range === '3M') return fullSeries.slice(-90);
    if (range === '6M') return fullSeries.slice(-180);
    if (range === 'YTD') return fullSeries.filter((d) => d.isYTD);
    return fullSeries;
  }, [range, fullSeries]);

  const latest = fullSeries[fullSeries.length - 1];
  const prevDay = fullSeries[fullSeries.length - 2];
  const first = fullSeries[0];
  const todayChange = latest.value - prevDay.value;
  const todayPct = (todayChange / prevDay.value) * 100;
  const inceptionPct = ((latest.value - first.value) / first.value) * 100;

  const buyCount = assetUniverse.filter((a) => getSignal(a.predictedReturn) === 'BUY').length;
  const sellCount = assetUniverse.filter((a) => getSignal(a.predictedReturn) === 'SELL').length;
  const avgConviction = assetUniverse.length > 0 ? Math.round(assetUniverse.reduce((s, a) => s + a.conviction, 0) / assetUniverse.length) : 0;

  const kpis = [
    { label: 'Portfolio Value', value: inr(latest.value), sub: `${inceptionPct >= 0 ? '+' : ''}${inceptionPct.toFixed(1)}% since inception`, up: inceptionPct >= 0, icon: Wallet, color: 'blue' },
    { label: "Today's P&L", value: `${todayChange >= 0 ? '+' : ''}${inr(todayChange)}`, sub: `${todayPct >= 0 ? '+' : ''}${todayPct.toFixed(2)}% vs prev. close`, up: todayChange >= 0, icon: Activity, color: 'emerald' },
    { label: 'Avg. Model Conviction', value: `${avgConviction}`, sub: 'across 10-asset universe', up: avgConviction >= 60, icon: Target, color: 'violet' },
    { label: 'Active Signals', value: `${buyCount} BUY`, sub: `${sellCount} SELL · ${assetUniverse.length - buyCount - sellCount} HOLD`, up: buyCount >= sellCount, icon: Zap, color: 'amber' },
    { label: 'Sharpe Ratio (PPO)', value: '1.84', sub: 'annualized, backtested', up: true, icon: TrendingUp, color: 'cyan' },
  ];

  return (
    <div className="min-h-screen w-full bg-slate-950 text-slate-100 flex font-sans">
      <aside className="hidden lg:flex lg:flex-col w-64 shrink-0 border-r border-slate-800 bg-slate-950/60 px-4 py-6 gap-6">
        <div className="flex items-center gap-2 px-2">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-cyan-400 flex items-center justify-center font-bold text-slate-950 text-sm">TA</div>
          <span className="font-semibold text-slate-100 tracking-tight text-sm">TradeAlpha</span>
        </div>
        <nav className="flex flex-col gap-1">
          <button onClick={() => setActiveView('overview')} className={`flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium ${activeView === 'overview' ? 'bg-blue-500/10 text-blue-400 border border-blue-500/20' : 'text-slate-400 hover:bg-slate-800/60 hover:text-slate-200'}`}><LayoutDashboard className="w-4 h-4" /> Dashboard Overview</button>
          <button onClick={() => setActiveView('opportunities')} className={`flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium ${activeView === 'opportunities' ? 'bg-blue-500/10 text-blue-400 border border-blue-500/20' : 'text-slate-400 hover:bg-slate-800/60 hover:text-slate-200'}`}><Compass className="w-4 h-4" /> Market Opportunities</button>
        </nav>
        <div className="mt-auto bg-slate-900/80 border border-slate-800 rounded-xl p-4 flex flex-col gap-3">
          <p className="text-xs font-semibold text-slate-400 uppercase tracking-wide">Terminal Controls</p>
          <div className="flex items-center justify-between text-xs"><span className="text-slate-500">Universe</span><span className="text-slate-300 font-medium">{assetUniverse.length} NSE assets</span></div>
          <div className="flex items-center justify-between text-xs"><span className="text-slate-500">Last sync</span><span className="text-slate-300 font-medium tabular-nums">{lastSync}</span></div>
          <button onClick={fetchMarketData} disabled={isRunning} className="w-full flex items-center justify-center gap-2 bg-blue-600 hover:bg-blue-500 disabled:bg-slate-700 text-white text-xs font-semibold py-2.5 rounded-lg transition-colors">
            {isRunning ? <><Loader2 className="w-3.5 h-3.5 animate-spin" /> Fetching Live Data…</> : <><Zap className="w-3.5 h-3.5" /> Run XGBoost + PPO</>}
          </button>
        </div>
      </aside>

      <main className="flex-1 min-w-0 px-4 sm:px-6 lg:px-8 py-6 flex flex-col gap-5 overflow-y-auto">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-2">
          <div>
            <h1 className="text-slate-100 font-bold text-2xl tracking-tight">{activeView === 'overview' ? 'Dashboard Overview' : 'Market Opportunities'}</h1>
            <p className="text-slate-500 text-sm mt-0.5">Decoupled React Frontend + FastAPI ML Backend</p>
          </div>
          <div className="flex items-center gap-2 text-xs text-emerald-400 bg-emerald-400/10 border border-emerald-400/20 rounded-full px-3 py-1.5 font-medium">
            <span className="relative flex h-2 w-2"><span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" /><span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500" /></span>
            API ONLINE · synced {lastSync}
          </div>
        </div>

        {activeView === 'overview' ? (
          <div className="flex flex-col gap-5">
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
              {kpis.map((k) => (
                <GlassCard key={k.label} className="p-5">
                  <div className="text-slate-500 text-xs mb-1">{k.label}</div>
                  <div className="text-slate-100 text-xl font-semibold tabular-nums">{k.value}</div>
                  <div className={`text-xs mt-1 flex items-center gap-1 ${k.up ? 'text-emerald-400' : 'text-rose-400'}`}>{k.up ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}{k.sub}</div>
                </GlassCard>
              ))}
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
              <GlassCard className="lg:col-span-2 p-5">
                <div className="flex justify-between items-center mb-4">
                  <h3 className="text-slate-100 font-semibold text-sm">Portfolio Growth</h3>
                  <div className="flex gap-1">{RANGES.map(r => <button key={r} onClick={() => setRange(r)} className={`px-2 py-1 text-xs rounded ${range === r ? 'bg-blue-600 text-white' : 'text-slate-400'}`}>{r}</button>)}</div>
                </div>
                <ResponsiveContainer width="100%" height={300}>
                  <AreaChart data={visibleSeries} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
                    <defs><linearGradient id="portfolioFill" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="#3b82f6" stopOpacity={0.4} /><stop offset="95%" stopColor="#3b82f6" stopOpacity={0} /></linearGradient></defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                    <XAxis dataKey="label" tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} minTickGap={40} />
                    <YAxis tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} tickFormatter={inrCompact} domain={['dataMin-10000', 'dataMax+10000']} />
                    <RechartsTooltip content={({ active, payload, label }) => {
                        if (!active || !payload || !payload.length) return null;
                        return <div className="bg-slate-900 border border-slate-700 p-2 rounded text-sm"><p className="text-slate-400 mb-1">{label}</p><p className="text-white font-bold">{inr(payload[0].value)}</p></div>;
                    }} />
                    <Area type="monotone" dataKey="value" stroke="#3b82f6" strokeWidth={2} fill="url(#portfolioFill)" activeDot={{ r: 4 }} />
                  </AreaChart>
                </ResponsiveContainer>
              </GlassCard>

              <div className="flex flex-col gap-5">
                <GlassCard className="p-5 flex-1">
                  <h3 className="text-slate-100 font-semibold text-sm mb-4">PPO Target Weights</h3>
                  <ResponsiveContainer width="100%" height={200}>
                    <PieChart>
                      <Pie data={assetUniverse} dataKey="targetWeight" nameKey="ticker" innerRadius={50} outerRadius={80} paddingAngle={2} stroke="none">
                        {assetUniverse.map((entry, i) => <Cell key={entry.ticker} fill={DONUT_COLORS[i % DONUT_COLORS.length]} />)}
                      </Pie>
                      <RechartsTooltip content={({ active, payload }) => {
                        if (!active || !payload?.length) return null;
                        return <div className="bg-slate-900 border border-slate-700 p-2 rounded"><p className="text-white font-bold text-sm">{payload[0].payload.ticker}</p><p className="text-slate-400 text-xs">{payload[0].payload.targetWeight}% allocation</p></div>;
                      }} />
                    </PieChart>
                  </ResponsiveContainer>
                </GlassCard>
              </div>
            </div>
          </div>
        ) : (
          <div className="flex flex-col gap-5">
            <GlassCard className="overflow-hidden">
              <table className="w-full text-left border-collapse text-sm">
                <thead>
                  <tr className="bg-slate-900 border-b border-slate-800 text-slate-400 text-xs uppercase">
                    <th className="p-4 font-semibold">Asset</th>
                    <th className="p-4 font-semibold">Price</th>
                    <th className="p-4 font-semibold">Pred. Return</th>
                    <th className="p-4 font-semibold">Conviction</th>
                    <th className="p-4 font-semibold">Signal</th>
                    <th className="p-4 font-semibold">Target Wt.</th>
                  </tr>
                </thead>
                <tbody>
                  {assetUniverse.sort((a,b) => b.conviction - a.conviction).map(r => {
                    const signal = getSignal(r.predictedReturn);
                    const { classes, Icon } = SIGNAL_META[signal];
                    const { bar, text } = convictionColor(r.conviction);
                    return (
                      <tr key={r.ticker} className="border-b border-slate-800/50 hover:bg-slate-800/30 transition-colors">
                        <td className="p-4"><p className="font-semibold text-white">{r.ticker}</p><p className="text-xs text-slate-500">{r.sector}</p></td>
                        <td className="p-4 text-slate-200 tabular-nums">{inr(r.price, 2)}</td>
                        <td className={`p-4 font-medium tabular-nums ${r.predictedReturn >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                          {r.predictedReturn >= 0 ? '+' : ''}{r.predictedReturn.toFixed(2)}%
                        </td>
                        <td className="p-4">
                          <div className="flex items-center gap-2 w-24">
                            <div className="flex-1 h-1.5 bg-slate-800 rounded-full"><div className={`h-full rounded-full ${bar}`} style={{ width: `${r.conviction}%` }} /></div>
                            <span className={`text-xs font-semibold ${text}`}>{r.conviction}</span>
                          </div>
                        </td>
                        <td className="p-4"><span className={`inline-flex items-center gap-1 px-2 py-1 rounded-full text-[10px] font-bold border ${classes}`}><Icon className="w-3 h-3" />{signal}</span></td>
                        <td className="p-4 text-slate-300 tabular-nums">{r.targetWeight.toFixed(1)}%</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
              {assetUniverse.length === 0 && <div className="p-10 text-center text-slate-500">Awaiting Market Data... Click 'Run XGBoost + PPO' to fetch.</div>}
            </GlassCard>
          </div>
        )}
      </main>
    </div>
  );
}