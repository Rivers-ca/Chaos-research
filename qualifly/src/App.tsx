import React, { useState } from 'react';

import {
  BrowserRouter,
  Routes,
  Route,
  Link,
  useParams,
  useNavigate
} from 'react-router-dom';

import {
  Briefcase,
  MessageSquare,
  ShieldCheck,
  Lock,
  GraduationCap,
  DollarSign,
  TrendingUp,
  Activity,
  Terminal,
  BarChart3,
  FileText,
  Check,
  X,
  Send,
  CheckCircle,
  Download,
  Loader2,
  Target,
  Database,
  Zap,
  Bird,
} from 'lucide-react';

import { AppProvider, useAppContext } from './AppContext';

import {
  MOCK_JOBS,
  MOCK_MATCHES,
  MOCK_THREADS,
  MOCK_SALARIES,
  MOCK_RESUMES,
  SCHOOLS,
  MAJORS,
  CLUBS
} from './data'


const Layout = ({ children }: { children: React.ReactNode }) => {
  const { role, cycleRole } = useAppContext();
  return (
    <div className="min-h-screen flex flex-col font-sans text-[#355872] bg-[#F7F8F0]">
      <header className="bg-white border-b sticky top-0 z-20 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-3 flex justify-between items-center">
          <Link to="/" className="text-xl font-extrabold text-[#355872] flex items-center gap-2 tracking-tight">
            <Bird className="w-7 h-7" /> QUALIFLY
          </Link>
          <nav className="flex gap-6 items-center font-medium text-sm text-gray-600">
            <Link to="/jobs" className="hover:text-[#7AAACE] flex items-center gap-1"><Briefcase className="w-4 h-4"/> Jobs</Link>
            <Link to="/forum" className="hover:text-[#7AAACE] flex items-center gap-1"><MessageSquare className="w-4 h-4"/> Forum</Link>
            <Link to="/salaries" className="hover:text-[#7AAACE] flex items-center gap-1"><DollarSign className="w-4 h-4"/> Salaries</Link>
            <Link to="/match" className="hover:text-[#7AAACE] flex items-center gap-1"><Zap className="w-4 h-4 text-amber-500"/> Matchmaking</Link>
            
            <div className="border-l pl-6 flex items-center gap-4">
            <Link to="/vault" className="hover:text-[#7AAACE] flex items-center gap-1 text-sm">
              <FileText className="w-4 h-4"/> Vault
            </Link>

            <Link to="/diagnostics" className="hover:text-[#7AAACE] flex items-center gap-1 text-sm">
              <Activity className="w-4 h-4"/> Diagnostics
            </Link>

            <button
              onClick={cycleRole}
              className={`px-3 py-1.5 rounded-md font-bold text-xs uppercase tracking-wide transition-colors ${
                role === 'Pro'
                  ? 'bg-amber-100 text-amber-700 border border-amber-300'
                  : role === 'Alumni'
                  ? ' text-[#7AAACE]'
                  : 'bg-gray-200 text-gray-700'
              }`}
            >
              {role} View
            </button>
          </div>
          </nav>
        </div>
      </header>
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 py-8">{children}</main>
    </div>
  );
};

// --- SHARED SIDEBAR ---
// --- SHARED SIDEBAR & HOOK ---
const useFilters = () => {
  const [selectedSchools, setSelectedSchools] = useState<string[]>([]);
  const [selectedMajors, setSelectedMajors] = useState<string[]>([]);
  const [selectedClubs, setSelectedClubs] = useState<string[]>([]);

  const toggleFilter = (item: string, type: 'school' | 'major' | 'club') => {
    if (type === 'school') setSelectedSchools(p => p.includes(item) ? p.filter(i => i !== item) : [...p, item]);
    if (type === 'major') setSelectedMajors(p => p.includes(item) ? p.filter(i => i !== item) : [...p, item]);
    if (type === 'club') setSelectedClubs(p => p.includes(item) ? p.filter(i => i !== item) : [...p, item]);
  };

  return { selectedSchools, selectedMajors, selectedClubs, toggleFilter };
};

const FilterSidebar = ({ filters }: { filters: ReturnType<typeof useFilters> }) => {
  const { selectedSchools, selectedMajors, selectedClubs, toggleFilter } = filters;

  const CustomCheckbox = ({ label, type, isSelected }: { label: string, type: 'school' | 'major' | 'club', isSelected: boolean }) => (
    <label className="flex items-center gap-3 mb-2 text-sm text-gray-600 cursor-pointer group">
      <div className={`w-4 h-4 rounded-[3px] border flex items-center justify-center transition-all ${
        isSelected 
          ? 'bg-[#355872] border-[#355872] text-white shadow-inner' 
          : 'bg-zinc-50 border-zinc-300 group-hover:border-[#7AAACE]'
      }`}>
        {isSelected && <Check className="w-3 h-3" />}
      </div>
      <span className={`transition-colors ${isSelected ? 'font-bold text-gray-900' : 'group-hover:text-[#7AAACE]'}`}>
        {label}
      </span>
    </label>
  );

  return (
    <div className="bg-white p-6 rounded-2xl border shadow-sm h-fit sticky top-24 space-y-6">
      <div className="flex items-center gap-2 mb-4">
        <Activity className="w-4 h-4 text-[#7AAACE]" />
        <h2 className="font-black text-xs uppercase tracking-widest text-gray-400">Filter</h2>
      </div>
      
      <div>
        <h3 className="font-bold mb-3 text-gray-900 text-sm border-b pb-2">University</h3>
        {SCHOOLS.map(s => (
          <div key={s} onClick={() => toggleFilter(s, 'school')}>
            <CustomCheckbox label={s} type="school" isSelected={selectedSchools.includes(s)} />
          </div>
        ))}
      </div>
      
      <div>
        <h3 className="font-bold mb-3 text-gray-900 text-sm border-b pb-2">Major</h3>
        {MAJORS.map(m => (
          <div key={m} onClick={() => toggleFilter(m, 'major')}>
            <CustomCheckbox label={m} type="major" isSelected={selectedMajors.includes(m)} />
          </div>
        ))}
      </div>
      
      <div>
        <h3 className="font-bold mb-3 text-gray-900 text-sm border-b pb-2">Clubs</h3>
        {CLUBS.map(c => (
          <div key={c} onClick={() => toggleFilter(c, 'club')}>
            <CustomCheckbox label={c} type="club" isSelected={selectedClubs.includes(c)} />
          </div>
        ))}
      </div>
    </div>
  );
};

// --- FORUM ---
const Forum = () => {
  const { threads } = useAppContext();

  console.log("THREAD COUNT:", threads.length);
  console.log("THREADS:", threads);

  const [activeTier, setActiveTier] = useState("All");
  const filters = useFilters(); // Initialize filters

  // Apply filter logic
  const filteredThreads = threads.filter(thread => {
    const matchTier = activeTier === 'All' || thread.tier === activeTier;
    const matchSchool = filters.selectedSchools.length === 0 || filters.selectedSchools.includes(thread.school);
    // Note: Assuming your mock thread data eventually has major/club properties. 
    // If not, just filter by school for now!
    return matchTier && matchSchool; 
  });

  return (
    <div className="grid lg:grid-cols-4 gap-8">
      <div className="lg:col-span-1">
        {/* Pass the hook state to the sidebar */}
        <FilterSidebar filters={filters} />
      </div>
      <div className="lg:col-span-3 space-y-6">
        <div className="flex gap-1 bg-white p-1 rounded-lg border shadow-sm w-fit">
          {['All', 'Alumni Insights', 'Student Hub', 'Job Discussions'].map(tier => (
            <button key={tier} onClick={() => setActiveTier(tier)} className={`px-4 py-2 rounded-md text-xs font-bold transition-colors ${activeTier === tier ? 'bg-[#355872] text-white shadow-md' : 'text-gray-500 hover:bg-gray-100'}`}>
              {tier}
            </button>
          ))}
        </div>
        
        {/* Show empty state if filters are too strict */}
        {filteredThreads.length === 0 && (
          <div className="text-center py-20 bg-white rounded-2xl border border-dashed">
            <p className="text-gray-400 font-medium">No threads match your exact filter.</p>
            <button onClick={() => window.location.reload()} className="text-[#7AAACE] text-sm font-bold mt-2 hover:underline">Clear Filters</button>
          </div>
        )}

        <div className="grid gap-4">
          {filteredThreads.map(thread => (
            <Link to={`/forum/${thread.id}`} key={thread.id} className="bg-white p-6 rounded-xl border shadow-sm hover:ring-2 hover:ring-blue-100 transition-all group">
              <div className="flex gap-2 mb-3">
                <span className="text-[10px] font-black px-2 py-0.5 rounded uppercase tracking-widest bg-purple-100 text-purple-700">{thread.tier}</span>
                <span className="text-[10px] font-black px-2 py-0.5 rounded uppercase tracking-widest bg-gray-100 text-gray-600">{thread.school}</span>
              </div>
              <h3 className="text-xl font-bold group-hover:text-[#7AAACE] transition-colors">{thread.title}</h3>
              <div className="mt-4 flex items-center gap-3 text-xs">
                <span className="font-bold">{thread.author}</span>
                {thread.isAlumni && <span className="flex items-center gap-1 text-[#7AAACE] font-bold bg-[#9CD5FF]/30 px-2 py-0.5 rounded border border-[#9CD5FF]/40"><ShieldCheck className="w-3 h-3"/> Alumni</span>}
                <span className="text-gray-300 ml-auto">{thread.comments.length} comments</span>
              </div>
            </Link>
          ))}
        </div>
      </div>
    </div>
  );
};

// --- THREAD DETAIL ---
const ThreadDetail = () => {
  const { id } = useParams();
  const { threads, role } = useAppContext();
  const thread = threads.find(t => t.id === id);
  if (!thread) return <div>Not found</div>;
  const isPaywalled = role !== 'Pro' && role !== 'Alumni';

  return (
    <div className="max-w-3xl mx-auto space-y-6">
      <Link to="/forum" className="text-sm font-bold text-[#7AAACE] flex items-center gap-1">← Back to Forums</Link>
      <div className="bg-white p-8 rounded-2xl border shadow-sm">
        <h1 className="text-3xl font-black mb-6">{thread.title}</h1>
        <p className="text-gray-700 leading-relaxed whitespace-pre-line">{thread.body}</p>
      </div>
      <div className="space-y-4">
        <h3 className="font-bold text-lg px-2 text-gray-400 uppercase tracking-widest">Discussion</h3>
        {thread.comments.map((c, i) => (
          <div key={c.id} className={`bg-white p-5 rounded-xl border shadow-sm ${(isPaywalled && i >= 2) ? 'blur-md opacity-30 select-none pointer-events-none' : ''}`}>
             <div className="flex items-center gap-2 mb-2 font-bold text-sm">
               {c.author} {c.isAlumni && <ShieldCheck className="w-3 h-3 text-blue-500"/>}
             </div>
             <p className="text-sm text-gray-600">{c.text}</p>
          </div>
        ))}
        {isPaywalled && (
          <div className="text-center py-10 bg-[#9CD5FF]/30 rounded-2xl border-2 border-dashed border-blue-200">
            <Lock className="w-8 h-8 mx-auto text-[#7AAACE] mb-2"/>
            <p className="font-bold">Upgrade to Qualifly Pro</p>
            <p className="text-xs text-gray-500 mb-4">Read full verified alumni discussions.</p>
            <button className="bg-[#355872] text-white px-6 py-2 rounded-lg font-bold text-sm">Unlock Discussion</button>
          </div>
        )}
      </div>
    </div>
  );
};

// --- MATCHMAKING (WITH PAYWALL GATE & INTRO MODAL) ---
const Matchmaking = () => {
  const { role } = useAppContext();
  const filters = useFilters(); // Reusing our sidebar hook
  
  // Modal State
  const [activeMatch, setActiveMatch] = useState<any | null>(null);
  const [isSending, setIsSending] = useState(false);
  const [isSent, setIsSent] = useState(false);

  const handleSendRequest = () => {
    setIsSending(true);
    // Simulate API call delay
    setTimeout(() => {
      setIsSending(false);
      setIsSent(true);
      // Auto-close after success
      setTimeout(() => {
        setActiveMatch(null);
        setIsSent(false);
      }, 2000);
    }, 1500);
  };

  const isLocked = role !== 'Pro' && role !== 'Alumni';

  if (isLocked) {
    return (
      <div className="max-w-4xl mx-auto py-12 animate-in fade-in zoom-in duration-500">
        <div className="bg-zinc-900 border-2 border-zinc-800 rounded-3xl p-12 text-center relative overflow-hidden shadow-2xl">
          <div className="absolute inset-0 bg-[linear-gradient(to_right,#27272a_1px,transparent_1px),linear-gradient(to_bottom,#27272a_1px,transparent_1px)] bg-[size:1.5rem_1.5rem] opacity-20"></div>
          <div className="relative z-10">
            <div className="inline-flex items-center justify-center w-20 h-20 bg-zinc-800 rounded-2xl mb-8 border border-zinc-700 shadow-inner">
              <Lock className="w-10 h-10 text-[#7AAACE] animate-pulse" />
            </div>
            <h2 className="text-4xl font-black text-white mb-4 tracking-tighter uppercase italic">
              Access Restricted // <span className="text-[#7AAACE]">Match Engine</span>
            </h2>
            <p className="text-zinc-400 max-w-lg mx-auto mb-10 font-medium leading-relaxed">
              Direct matchmaking with verified alumni is a <span className="text-white font-bold">Pro-tier diagnostic tool</span>. Upgrade your clearance level to bypass standard application queues.
            </p>
            <div className="grid md:grid-cols-2 gap-4 max-w-md mx-auto mb-10 text-left">
              <div className="flex items-center gap-3 text-[10px] font-mono text-zinc-300 bg-zinc-800/50 p-3 rounded-xl border border-zinc-700">
                <Zap className="w-4 h-4 text-[#7AAACE]" /> AI-POWERED MATCHING
              </div>
              <div className="flex items-center gap-3 text-[10px] font-mono text-zinc-300 bg-zinc-800/50 p-3 rounded-xl border border-zinc-700">
                <ShieldCheck className="w-4 h-4 text-[#7AAACE]" /> VERIFIED HANDSHAKES
              </div>
            </div>
            <button className="bg-[#7AAACE] text-zinc-900 px-10 py-4 rounded-xl font-black text-xs uppercase tracking-widest hover:bg-[#9CD5FF] hover:scale-105 transition-all shadow-[0_0_20px_rgba(122,170,206,0.4)]">
              Unlock Qualifly Pro
            </button>
            <p className="mt-8 font-mono text-[10px] text-zinc-600 uppercase tracking-[0.3em]">
              Sign up to become a quailty {role.toUpperCase()}
            </p>
          </div>
        </div>
      </div>
    );
  }

  // Filter logic applied to Matches
  const filteredMatches = MOCK_MATCHES.filter(match => {
    return filters.selectedSchools.length === 0 || filters.selectedSchools.includes(match.school);
  });

  return (
    <>
      <div className="grid lg:grid-cols-4 gap-8 animate-in slide-in-from-bottom-4 duration-700">
        <div className="lg:col-span-1"><FilterSidebar filters={filters} /></div>
        
        <div className="lg:col-span-3">
          <div className="flex justify-between items-end mb-6">
            <h2 className="text-2xl font-black flex items-center gap-2 uppercase tracking-tight text-gray-900">
              <Zap className="w-6 h-6 text-amber-500"/> Alumni Network
            </h2>
            <span className="text-xs font-bold text-gray-400">{filteredMatches.length} Matches Found</span>
          </div>

          {filteredMatches.length === 0 ? (
             <div className="text-center py-20 bg-white rounded-2xl border border-dashed">
               <p className="text-gray-400 font-medium">No alumni match your current filter.</p>
             </div>
          ) : (
            <div className="grid md:grid-cols-2 gap-6">
              {filteredMatches.map(match => (
                <div key={match.id} className="bg-white p-6 rounded-2xl border shadow-sm relative overflow-hidden flex flex-col items-center text-center group hover:shadow-xl hover:-translate-y-1 transition-all">
                  <div className="absolute top-4 right-4 bg-green-50 text-green-700 px-3 py-1 rounded-full text-[10px] font-black border border-green-200 flex items-center gap-1">
                    <Zap className="w-3 h-3 fill-current"/> {match.score}% MATCH
                  </div>
                  
                  <div className="w-20 h-20 bg-gradient-to-br from-zinc-800 to-zinc-600 rounded-full mb-4 flex items-center justify-center text-white font-black text-2xl shadow-lg group-hover:scale-110 transition-transform">
                    {match.name[0]}
                  </div>
                  
                  <h3 className="text-xl font-bold text-gray-900 mb-1">{match.name}</h3>
                  <p className="text-sm text-[#7AAACE] font-bold uppercase tracking-tight mb-4">{match.role} @ {match.company}</p>
                  
                  <div className="flex flex-wrap gap-1.5 justify-center mb-6">
                    <span className="text-[10px] bg-gray-100 text-gray-600 px-2.5 py-1 rounded-full font-bold border border-gray-200">{match.school}</span>
                    <span className="text-[10px] bg-gray-100 text-gray-600 px-2.5 py-1 rounded-full font-bold border border-gray-200">{match.club}</span>
                  </div>

                  <div className="w-full pt-4 border-t flex gap-2">
                    <button 
                      onClick={() => setActiveMatch(match)}
                      className="flex-1 bg-zinc-900 text-white py-2.5 rounded-xl font-bold text-xs hover:bg-[#355872] transition shadow-md"
                    >
                      Request Intro
                    </button>
                    <button className="px-3 bg-zinc-50 text-gray-400 border rounded-xl hover:text-[#7AAACE] hover:bg-white transition">
                      <MessageSquare className="w-4 h-4"/>
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* THE MODAL OVERLAY */}
      {activeMatch && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-zinc-900/60 backdrop-blur-sm animate-in fade-in duration-200">
          <div className="bg-white rounded-3xl shadow-2xl w-full max-w-lg overflow-hidden animate-in zoom-in-95 duration-200">
            
            {/* Modal Header */}
            <div className="bg-zinc-900 p-6 text-white flex justify-between items-center relative">
              {/* Decorative grid */}
              <div className="absolute inset-0 bg-[linear-gradient(to_right,#3f3f46_1px,transparent_1px),linear-gradient(to_bottom,#3f3f46_1px,transparent_1px)] bg-[size:1rem_1rem] opacity-20"></div>
              
              <div className="relative z-10 flex items-center gap-3">
                <div className="w-10 h-10 bg-zinc-800 rounded-full flex items-center justify-center font-black border border-zinc-700">
                  {activeMatch.name[0]}
                </div>
                <div>
                  <h3 className="font-black text-lg leading-tight">{activeMatch.name}</h3>
                  <p className="text-[10px] text-[#7AAACE] font-mono uppercase tracking-widest">{activeMatch.role} // {activeMatch.company}</p>
                </div>
              </div>
              <button 
                onClick={() => !isSending && setActiveMatch(null)}
                className="relative z-10 text-zinc-400 hover:text-white transition-colors"
              >
                <X className="w-6 h-6" />
              </button>
            </div>

            {/* Modal Body */}
            <div className="p-6 space-y-4">
              <div className="flex items-center justify-between text-xs font-bold text-gray-400 uppercase tracking-widest border-b pb-2">
                <span>Subject: Connecting via Qualifly</span>
                <span className="text-green-600 flex items-center gap-1"><ShieldCheck className="w-3 h-3"/> High Match</span>
              </div>
              
              <textarea 
                className="w-full h-48 p-4 bg-zinc-50 border border-zinc-200 rounded-xl text-sm text-gray-700 focus:outline-none focus:ring-2 focus:ring-[#7AAACE]/50 resize-none font-medium leading-relaxed"
                defaultValue={`Hi ${activeMatch.name.split(' ')[0]},\n\nI noticed we both were involved in ${activeMatch.club} at ${activeMatch.school}. I'm currently exploring opportunities in the ${activeMatch.company} ecosystem and would love to hear your perspective on the culture there.\n\nDo you have 10 minutes for a quick chat next week?\n\nBest,\n[Your Name]`}
              />

              {/* Modal Footer */}
              <div className="pt-2 flex justify-end gap-3">
                <button 
                  onClick={() => setActiveMatch(null)}
                  disabled={isSending || isSent}
                  className="px-5 py-2.5 text-xs font-bold text-gray-500 hover:text-gray-800 transition-colors disabled:opacity-50"
                >
                  Cancel
                </button>
                <button 
                  onClick={handleSendRequest}
                  disabled={isSending || isSent}
                  className={`flex items-center gap-2 px-6 py-2.5 rounded-xl font-bold text-xs transition-all shadow-md ${
                    isSent 
                      ? 'bg-green-500 text-white' 
                      : 'bg-[#355872] text-white hover:bg-[#2c4b61] hover:-translate-y-0.5'
                  }`}
                >
                  {isSending ? (
                    <><Activity className="w-4 h-4 animate-spin"/> Transmitting...</>
                  ) : isSent ? (
                    <><CheckCircle className="w-4 h-4"/> Request Sent</>
                  ) : (
                    <><Send className="w-4 h-4"/> Send Request</>
                  )}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
};
// --- SALARIES ---
const SalaryExplorer = () => {
  const filters = useFilters();

  // Apply filter logic to the salaries
  const filteredSalaries = MOCK_SALARIES.filter(s => {
    const matchSchool = filters.selectedSchools.length === 0 || filters.selectedSchools.includes(s.school);
    const matchMajor = filters.selectedMajors.length === 0 || filters.selectedMajors.includes(s.major);
    // You can also add club filtering here if your data supports it
    return matchSchool && matchMajor;
  });

  return(
    <div className="grid lg:grid-cols-4 gap-8 animate-in slide-in-from-bottom-4 duration-700">
      <div className="lg:col-span-1"><FilterSidebar filters={filters} /></div>
      
      <div className="lg:col-span-3 space-y-6">
        
        {/* Header Banner */}
        <div className="bg-[#355872] p-8 rounded-2xl text-white shadow-lg relative overflow-hidden">
          {/* Subtle grid background for the terminal aesthetic */}
          <div className="absolute inset-0 bg-[linear-gradient(to_right,#ffffff15_1px,transparent_1px),linear-gradient(to_bottom,#ffffff15_1px,transparent_1px)] bg-[size:1.5rem_1.5rem]"></div>
          
          <div className="relative z-10">
            <h1 className="text-3xl font-black mb-2 flex items-center gap-2 italic tracking-tight">
              QUALIFLY <TrendingUp className="w-8 h-8 not-italic text-[#9CD5FF]"/> DATA
            </h1>
            <p className="font-medium opacity-80 text-sm">Salary transparency verified through university club networks and offer letters.</p>
          </div>
        </div>

        <div className="flex justify-between items-center px-2">
          <h2 className="text-sm font-bold text-gray-500 uppercase tracking-widest">Verified Compensation Data</h2>
          <span className="text-xs font-bold text-[#355872] bg-[#9CD5FF]/30 px-3 py-1 rounded-full border border-[#9CD5FF]/50">
            {filteredSalaries.length} Records
          </span>
        </div>

        {/* The Filter Gate / Empty State */}
        {filteredSalaries.length === 0 ? (
          <div className="text-center py-20 bg-white rounded-2xl border border-dashed hover:border-[#7AAACE] transition-colors">
            <DollarSign className="w-10 h-10 mx-auto text-gray-300 mb-3" />
            <p className="text-gray-400 font-medium">No compensation data matches your current filter.</p>
          </div>
        ) : (
          <div className="grid gap-6">
            {filteredSalaries.map(s => (
              <div key={s.id} className="bg-white p-6 rounded-2xl border shadow-sm hover:shadow-md transition-all group">
                <div className="flex justify-between items-start mb-6 border-b border-gray-50 pb-4">
                  <div>
                    <h3 className="text-xl font-black group-hover:text-[#355872] transition-colors">{s.role}</h3>
                    <p className="text-xs font-bold text-gray-400 uppercase tracking-widest mt-1">{s.major} • {s.school}</p>
                  </div>
                  <div className="text-right">
                    <span className="text-3xl font-black text-green-600 tracking-tighter">${(s.medianPay/1000).toFixed(0)}k</span>
                    <p className="text-[10px] font-black text-gray-400 uppercase mt-1">Median 1st Year</p>
                  </div>
                </div>
                
                <div className="grid md:grid-cols-2 gap-8 items-center">
                  {/* Visual Spectrum Bar */}
                  <div>
                    <p className="text-[10px] font-black text-gray-400 uppercase mb-3">Pay Spectrum (Base + Bonus)</p>
                    <div className="h-3 bg-gray-100 rounded-full relative mb-2 overflow-hidden shadow-inner border border-gray-200">
                      <div className="absolute left-[15%] right-[15%] h-full bg-gradient-to-r from-[#9CD5FF] to-[#355872] shadow-[0_0_10px_rgba(59,130,246,0.5)] rounded-full"></div>
                    </div>
                    <div className="flex justify-between text-[10px] font-bold text-gray-500 px-1">
                      <span>Low: ${(s.lowPay/1000).toFixed(0)}k</span>
                      <span>High: ${(s.highPay/1000).toFixed(0)}k</span>
                    </div>
                  </div>
                  
                  {/* Outcomes Tags */}
                  <div className="bg-[#F7F8F0] p-4 rounded-xl border border-gray-200">
                    <p className="text-[10px] font-black text-gray-400 uppercase mb-2">Verified Placements</p>
                    <div className="flex flex-wrap gap-2">
                      {s.outcomes.map((o: string, idx: number) => (
                        <div key={idx} className="flex items-center gap-1.5 text-xs font-bold text-gray-700 bg-white px-2 py-1 rounded-lg shadow-sm border border-gray-100">
                          <ShieldCheck className="w-3 h-3 text-blue-500"/> {o}
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
// --- RESUME VAULT ---
const ResumeVault = () => {
  const { role } = useAppContext();
  const [activeResume, setActiveResume] = useState<any | null>(null);

  const isLocked = role !== 'Pro' && role !== 'Alumni';

  if (isLocked) {
    return (
      <div className="text-center py-20 max-w-2xl mx-auto animate-in fade-in zoom-in duration-500">
        <div className="bg-zinc-900 border-2 border-zinc-800 rounded-3xl p-12 relative overflow-hidden shadow-2xl">
          <div className="absolute inset-0 bg-[linear-gradient(to_right,#27272a_1px,transparent_1px),linear-gradient(to_bottom,#27272a_1px,transparent_1px)] bg-[size:1.5rem_1.5rem] opacity-20"></div>
          <div className="relative z-10">
            <Lock className="w-12 h-12 mx-auto text-[#7AAACE] mb-6 animate-pulse"/>
            <h2 className="text-3xl font-black mb-3 text-white uppercase tracking-tighter">Classified Data</h2>
            <p className="text-zinc-400 mb-8 font-medium">Unlock the exact, verified resumes that secured Tier-1 offers at Goldman Sachs, Bain, and Google. Pro access required.</p>
            <button className="bg-[#7AAACE] text-zinc-900 px-8 py-3 rounded-xl font-black text-xs uppercase tracking-widest hover:bg-[#9CD5FF] transition-all shadow-[0_0_15px_rgba(122,170,206,0.3)]">
              Upgrade to Pro
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="animate-in slide-in-from-bottom-4 duration-700">
      <div className="flex justify-between items-end mb-6">
        <h2 className="text-2xl font-black flex items-center gap-2 uppercase tracking-tight text-gray-900">
          <FileText className="w-6 h-6 text-[#7AAACE]"/> Verified Offer Resumes
        </h2>
      </div>

      <div className="grid md:grid-cols-3 gap-6">
        {MOCK_RESUMES.map(resume => (
          <div key={resume.id} className="bg-white p-6 rounded-2xl border shadow-sm hover:border-[#7AAACE]/50 hover:shadow-md transition-all group flex flex-col justify-between">
            <div>
              <div className="flex justify-between items-start mb-4">
                <div className="bg-[#F7F8F0] p-3 rounded-xl">
                  <FileText className="w-6 h-6 text-[#355872]"/>
                </div>
                <span className="text-[10px] font-black text-green-600 bg-green-50 px-2 py-1 rounded-full border border-green-200">
                  VERIFIED OFFER
                </span>
              </div>
              <h3 className="font-black text-lg leading-tight mb-1 group-hover:text-[#7AAACE] transition-colors">{resume.role}</h3>
              <p className="text-sm text-gray-500 font-bold mb-4">{resume.firm}</p>
              
              <div className="space-y-2 mb-6">
                <div className="flex justify-between text-xs border-b border-gray-50 pb-1">
                  <span className="text-gray-400">Target School</span>
                  <span className="font-bold text-gray-700">{resume.school}</span>
                </div>
                <div className="flex justify-between text-xs border-b border-gray-50 pb-1">
                  <span className="text-gray-400">Cumulative GPA</span>
                  <span className="font-bold text-gray-700">{resume.gpa}</span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-gray-400">Offer Year</span>
                  <span className="font-bold text-gray-700">{resume.year}</span>
                </div>
              </div>
            </div>
            
            <button 
              onClick={() => setActiveResume(resume)}
              className="w-full bg-zinc-900 text-white py-3 rounded-xl text-xs font-bold hover:bg-[#355872] transition-colors shadow-sm"
            >
              Inspect Document
            </button>
          </div>
        ))}
      </div>

      {/* PDF VIEWER MODAL */}
      {activeResume && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6 bg-zinc-900/80 backdrop-blur-sm animate-in fade-in duration-200">
          <div className="w-full max-w-4xl h-[90vh] bg-zinc-200 rounded-2xl overflow-hidden flex flex-col shadow-2xl border border-zinc-700 animate-in zoom-in-95 duration-200">
            
            {/* Viewer Toolbar */}
            <div className="bg-zinc-800 text-zinc-300 px-4 py-3 flex justify-between items-center border-b border-zinc-900">
              <div className="flex items-center gap-3">
                <FileText className="w-4 h-4 text-[#7AAACE]" />
                <span className="text-xs font-mono font-bold tracking-wider">
                  {activeResume.firm.replace(/\s+/g, '_')}_{activeResume.role.replace(/\s+/g, '_')}_Redacted.pdf
                </span>
              </div>
              <div className="flex items-center gap-2">
                <button className="p-1.5 hover:bg-zinc-700 rounded-md transition-colors text-zinc-400 hover:text-white" title="Download Source">
                  <Download className="w-4 h-4" />
                </button>
                <div className="w-px h-4 bg-zinc-600 mx-1"></div>
                <button 
                  onClick={() => setActiveResume(null)}
                  className="p-1.5 hover:bg-red-500/20 rounded-md transition-colors text-zinc-400 hover:text-red-400"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
            </div>

            {/* Document Container (Scrollable) */}
            <div className="flex-1 overflow-y-auto p-4 sm:p-8 flex justify-center custom-scrollbar">
              
              {/* The "Paper" */}
              <div className="bg-white w-full max-w-3xl min-h-[1056px] shadow-lg p-10 sm:p-16 text-zinc-900 font-serif leading-relaxed relative">
                
                {/* Watermark */}
                <div className="absolute inset-0 flex items-center justify-center opacity-[0.03] pointer-events-none select-none">
                  <span className="text-8xl font-black uppercase rotate-[-45deg] tracking-widest">QUALIFLY VERIFIED</span>
                </div>

                {/* Simulated Resume Content */}
                <div className="relative z-10">
                  <div className="text-center border-b-[1.5px] border-zinc-800 pb-4 mb-6">
                    <h1 className="text-3xl font-bold uppercase tracking-widest mb-2 blur-[4px] select-none">REDACTED CANDIDATE</h1>
                    <p className="text-xs font-sans text-zinc-600">
                      <span className="blur-[3px] select-none">123 Campus Drive, City, ST 12345</span> • <span className="blur-[3px] select-none">student@university.edu</span> • <span className="blur-[3px] select-none">(555) 123-4567</span>
                    </p>
                  </div>

                  <div className="mb-6">
                    <h2 className="text-sm font-bold uppercase tracking-widest border-b border-zinc-300 mb-3 pb-1">Education</h2>
                    <div className="flex justify-between items-baseline mb-1">
                      <h3 className="font-bold">{activeResume.school}</h3>
                      <span className="text-sm italic">May {activeResume.year}</span>
                    </div>
                    <p className="text-sm italic mb-2">Bachelor of Arts/Science (Mock Major)</p>
                    <ul className="list-disc list-inside text-sm space-y-1 ml-2">
                      <li><strong>Cumulative GPA:</strong> {activeResume.gpa} / 4.0</li>
                      <li><strong>Relevant Coursework:</strong> Financial Accounting, Corporate Finance, Data Structures, Microeconomics</li>
                      <li><strong>Honors:</strong> Dean's List (All Semesters), Target Club Leadership</li>
                    </ul>
                  </div>

                  <div className="mb-6">
                    <h2 className="text-sm font-bold uppercase tracking-widest border-b border-zinc-300 mb-3 pb-1">Experience</h2>
                    
                    <div className="mb-4">
                      <div className="flex justify-between items-baseline mb-1">
                        <h3 className="font-bold">{activeResume.firm}</h3>
                        <span className="text-sm italic">Summer {Number(activeResume.year) - 1}</span>
                      </div>
                      <p className="text-sm italic mb-2">Incoming {activeResume.role}</p>
                      <ul className="list-disc list-outside text-sm space-y-1.5 ml-5 text-zinc-700">
                        <li>Secured highly competitive offer after completing rigorous technical and behavioral interview process.</li>
                        <li>Selected as one of 40 summer analysts from an applicant pool of over 3,000 students.</li>
                      </ul>
                    </div>

                    <div className="mb-4">
                      <div className="flex justify-between items-baseline mb-1">
                        <h3 className="font-bold blur-[3px] select-none">Previous Tier-2 Firm</h3>
                        <span className="text-sm italic">Summer {Number(activeResume.year) - 2}</span>
                      </div>
                      <p className="text-sm italic mb-2 blur-[2px] select-none">Financial/Strategy Intern</p>
                      <ul className="list-disc list-outside text-sm space-y-1.5 ml-5 text-zinc-700">
                        <li>Developed advanced financial models in Excel, utilizing INDEX/MATCH and nested IF statements to forecast revenue growth.</li>
                        <li>Compiled market research spanning 50+ competitors to support a live M&A pitch for a $500M tech acquisition.</li>
                        <li>Presented weekly macro-economic trend summaries to managing directors and senior partners.</li>
                      </ul>
                    </div>
                  </div>

                  <div>
                    <h2 className="text-sm font-bold uppercase tracking-widest border-b border-zinc-300 mb-3 pb-1">Skills & Interests</h2>
                    <ul className="text-sm space-y-2">
                      <li><strong>Technical:</strong> Advanced Excel (Macros, VBA), Python (Pandas, NumPy), SQL, PowerPoint.</li>
                      <li><strong>Languages:</strong> English (Native), Spanish (Conversational).</li>
                      <li><strong>Interests:</strong> Competitive Rowing, European History, Intramural Basketball, Algorithmic Trading.</li>
                    </ul>
                  </div>
                </div>

              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// --- JOBS ---
const JobsExplorer = () => {
  const filters = useFilters();

  const filteredJobs = MOCK_JOBS.filter(job => {
    // Assuming your MOCK_JOBS have a 'schoolTarget' or similar property. 
    // Modify 'job.location' to whatever field makes sense to filter in your mock data!
    const matchSchool = filters.selectedSchools.length === 0 || filters.selectedSchools.some(s => job.location.includes(s) || job.company.includes(s));
    return matchSchool;
  });

  return (
    <div className="grid lg:grid-cols-4 gap-8">
      <div className="lg:col-span-1">
        <FilterSidebar filters={filters} />
      </div>
      <div className="lg:col-span-3 space-y-4">
        <div className="flex justify-between items-end mb-4">
          <h2 className="text-2xl font-black flex items-center gap-2 uppercase tracking-tight text-gray-900">
            <Briefcase className="w-6 h-6 text-[#7AAACE]"/> High-Signal Openings
          </h2>
          <span className="text-xs font-bold text-gray-400">{filteredJobs.length} Results</span>
        </div>

        {filteredJobs.length === 0 ? (
           <div className="text-center py-20 bg-white rounded-2xl border border-dashed">
             <p className="text-gray-400 font-medium">No roles match your current filter.</p>
           </div>
        ) : (
          filteredJobs.slice(0, 10).map(job => (
            <div key={job.id} className="bg-white p-6 rounded-2xl border shadow-sm hover:border-[#9CD5FF]/40 transition-all flex justify-between items-center group">
              <div>
                <h3 className="text-lg font-black group-hover:text-[#7AAACE] transition-colors">{job.title}</h3>
                <p className="text-sm font-bold text-gray-500 mb-3">{job.company} • {job.location}</p>
                <span className="text-[10px] font-black bg-[#9CD5FF]/30 text-[#7AAACE] px-3 py-1 rounded-full uppercase tracking-widest border border-[#9CD5FF]/40">Alumni Referred</span>
              </div>
              <button className="bg-gray-900 text-white px-6 py-2.5 rounded-xl font-bold text-sm hover:bg-[#355872] shadow-md transition-all">Apply</button>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

// --- DIAGNOSTICS ---
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer } from "recharts";

// Custom Tooltip for the Hardware Aesthetic
const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    return (
      <div className="bg-zinc-900 border border-zinc-700 p-4 rounded-xl shadow-2xl text-white font-mono text-xs min-w-[200px] z-50">
        <p className="text-[#7AAACE] font-bold mb-3 uppercase tracking-widest border-b border-zinc-700 pb-2">Year {label} Projection</p>
        <div className="space-y-2">
          <p className="flex justify-between gap-4"><span className="text-zinc-400">Total Comp:</span> <span className="font-bold">${data.totalComp.toLocaleString()}</span></p>
          <p className="flex justify-between gap-4"><span className="text-zinc-400">Debt Paid:</span> <span className="text-orange-400">-${data.debtPayment.toLocaleString()}</span></p>
          <p className="flex justify-between gap-4 pt-2 border-t border-zinc-800 mt-2"><span className="text-zinc-300 font-bold">Net Wealth:</span> <span className="font-bold text-green-400">${data.cumulative.toLocaleString()}</span></p>
        </div>
      </div>
    );
  }
  return null;
};

type CareerPath = "IB" | "PE" | "Consulting" | "CS";

const Diagnostics = () => {
  const [career, setCareer] = useState<CareerPath>("IB");
  const [years, setYears] = useState<number>(10);
  const [loanAmount, setLoanAmount] = useState<number>(50000);

  // Expanded Career Dictionary
  const careerModels: Record<CareerPath, { base: number, growth: number, bonus: number, equity: number, label: string }> = {
    IB: { base: 120000, growth: 0.12, bonus: 0.60, equity: 0, label: "Investment Banking" },
    PE: { base: 150000, growth: 0.15, bonus: 1.00, equity: 0, label: "Private Equity" }, // Simplified carry as high bonus
    Consulting: { base: 110000, growth: 0.14, bonus: 0.20, equity: 0, label: "Management Consulting" },
    CS: { base: 140000, growth: 0.10, bonus: 0.15, equity: 30000, label: "Software Engineering" }
  };

  const taxRate = 0.32;
  const { base, growth, bonus, equity } = careerModels[career];

  let salary = base;
  let cumulative = 0;
  let debtRemaining = loanAmount;
  const data = [];

  // Upgraded Data Loop to feed the custom tooltip
  for (let year = 1; year <= years; year++) {
    const yearlyBonus = salary * bonus;
    const totalComp = salary + yearlyBonus + equity;
    const afterTax = totalComp * (1 - taxRate);

    // Allocate 15% of after-tax to debt until gone
    const debtPayment = Math.min(afterTax * 0.15, debtRemaining);
    debtRemaining -= debtPayment;

    const netIncome = afterTax - debtPayment;
    cumulative += netIncome;

    data.push({
      year,
      totalComp: Math.round(totalComp),
      debtPayment: Math.round(debtPayment),
      cumulative: Math.round(cumulative)
    });

    salary *= (1 + growth);
  }

  return (
    <div className="space-y-8 animate-in fade-in duration-500 max-w-5xl mx-auto">
      
      {/* Header */}
      <div>
        <h2 className="text-2xl font-black flex items-center gap-2 uppercase tracking-tight text-gray-900 mb-2">
          <Activity className="w-6 h-6 text-[#7AAACE]"/> Wealth Trajectory Engine
        </h2>
        <p className="text-sm text-gray-500 font-medium">Model your 10-year post-grad liquid net worth based on verified industry progression data.</p>
      </div>

      <div className="grid md:grid-cols-3 gap-8">
        {/* Controls Sidebar */}
        <div className="md:col-span-1 space-y-8 bg-white p-6 rounded-2xl border shadow-sm h-fit">
          
          {/* Career Toggles */}
          <div>
            <label className="text-[10px] font-black text-gray-400 uppercase tracking-widest mb-3 block">Career Track</label>
            <div className="flex flex-col gap-2">
              {(Object.keys(careerModels) as CareerPath[]).map((c) => (
                <button
                  key={c}
                  onClick={() => setCareer(c)}
                  className={`px-4 py-2.5 rounded-lg font-bold text-xs text-left transition-all border ${
                    career === c
                      ? "bg-zinc-900 text-white border-zinc-900 shadow-md"
                      : "bg-zinc-50 text-zinc-600 border-zinc-200 hover:border-[#7AAACE] hover:text-[#7AAACE]"
                  }`}
                >
                  {careerModels[c].label}
                </button>
              ))}
            </div>
          </div>

          <hr className="border-zinc-100" />

          {/* Sliders */}
          <div>
            <div className="flex justify-between mb-2">
              <label className="text-[10px] font-black text-gray-400 uppercase tracking-widest">Time Horizon</label>
              <span className="text-xs font-bold text-[#7AAACE]">{years} Years</span>
            </div>
            <input
              type="range" min="1" max="20" value={years}
              onChange={(e) => setYears(Number(e.target.value))}
              className="w-full accent-[#355872]"
            />
          </div>

          <div>
            <div className="flex justify-between mb-2">
              <label className="text-[10px] font-black text-gray-400 uppercase tracking-widest">Starting Debt</label>
              <span className="text-xs font-bold text-orange-500">${loanAmount.toLocaleString()}</span>
            </div>
            <input
              type="range" min="0" max="400000" step="5000" value={loanAmount}
              onChange={(e) => setLoanAmount(Number(e.target.value))}
              className="w-full accent-[#355872]"
            />
          </div>
        </div>

        {/* Output & Graph Area */}
        <div className="md:col-span-2 space-y-6">
          {/* Wealth Output Card */}
          <div className="bg-zinc-900 text-white p-8 rounded-2xl border border-zinc-800 shadow-lg relative overflow-hidden">
             {/* Subtle background grid */}
            <div className="absolute inset-0 bg-[linear-gradient(to_right,#27272a_1px,transparent_1px),linear-gradient(to_bottom,#27272a_1px,transparent_1px)] bg-[size:1rem_1rem] opacity-30"></div>
            
            <div className="relative z-10 flex flex-col md:flex-row md:items-end justify-between gap-4">
              <div>
                <div className="text-[10px] font-black uppercase tracking-widest text-zinc-400 mb-1">
                  Projected Net Wealth After {years} Years
                </div>
                <div className="text-5xl md:text-6xl font-black text-transparent bg-clip-text bg-gradient-to-r from-green-400 to-[#7AAACE] tracking-tighter">
                  ${data[data.length - 1]?.cumulative.toLocaleString()}
                </div>
              </div>
              <div className="text-right">
                <div className="text-[10px] font-black uppercase tracking-widest text-zinc-500 mb-1">Debt Status</div>
                <div className={`text-xl font-bold ${debtRemaining > 0 ? 'text-orange-400' : 'text-zinc-300'}`}>
                  {debtRemaining > 0 ? `$${Math.round(debtRemaining).toLocaleString()} Remaining` : 'Cleared'}
                </div>
              </div>
            </div>
          </div>

          {/* Graph Card */}
          <div className="bg-white p-6 rounded-2xl border shadow-sm">
            <ResponsiveContainer width="100%" height={350}>
              <LineChart data={data} margin={{ top: 20, right: 20, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f4f4f5" vertical={false} />
                <XAxis 
                  dataKey="year" 
                  tick={{ fontSize: 12, fill: '#a1a1aa', fontWeight: 600 }} 
                  axisLine={false} 
                  tickLine={false}
                  tickFormatter={(v) => `Yr ${v}`}
                />
                <YAxis 
                  tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`} 
                  tick={{ fontSize: 12, fill: '#a1a1aa', fontWeight: 600 }}
                  axisLine={false}
                  tickLine={false}
                />
                <Tooltip content={<CustomTooltip />} cursor={{ stroke: '#e4e4e7', strokeWidth: 2, strokeDasharray: '4 4' }} />
                <Line
                  type="monotone"
                  dataKey="cumulative"
                  stroke="#355872"
                  strokeWidth={4}
                  dot={{ r: 4, fill: '#355872', strokeWidth: 2, stroke: '#fff' }}
                  activeDot={{ r: 8, fill: '#7AAACE', strokeWidth: 0 }}
                  animationDuration={1000}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
};

//landing
const Landing = () => {
  const navigate = useNavigate();
  const [loadingPath, setLoadingPath] = useState<string | null>(null);

  // Form State for the Trajectory Engine
  // (Safely defaulting to strings in case SCHOOLS/MAJORS aren't loaded yet)
  const [school, setSchool] = useState("Wharton"); 
  const [major, setMajor] = useState("Finance");
  const [gpa, setGpa] = useState("3.8");

  const handleAnalyze = (path: string) => {
    setLoadingPath(path);
    // Simulate the "Engine Crunching Data" for 1.2 seconds before routing
    setTimeout(() => {
      navigate(path);
    }, 1200);
  };

  return (
    <div className="flex flex-col w-full overflow-hidden">
      {/* 1. HERO SECTION */}
      <div className="relative min-h-[85vh] flex flex-col justify-center items-center text-center px-4 pt-10 pb-20">
        {/* Background Technical Grid */}
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#f4f4f5_1px,transparent_1px),linear-gradient(to_bottom,#f4f4f5_1px,transparent_1px)] bg-[size:1rem_2rem] -z-10"></div>
        
        {/* Top Badge */}
        <div className="font-mono text-[10px] font-bold tracking-[0.3em] text-[#7AAACE] uppercase mb-8 border-2 border-orange-200 bg-white px-4 py-1.5 rounded-full flex items-center gap-2 shadow-sm">
          <span className="w-2 h-2 bg-[#7AAACE] rounded-full animate-pulse"></span>
          System Active // V 1.0.4
        </div>

        <h1 className="text-6xl md:text-8xl lg:text-9xl font-black text-zinc-900 tracking-tighter leading-[0.9] mb-8">
          Quality Data.<br/>
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-[#7AAACE] to-[#9CD5FF]">Quality Outcomes.</span>
        </h1>

        <p className="text-xl text-zinc-600 max-w-2xl font-medium mb-12 leading-relaxed">
          The traditional job hunt is a black box. Qualifly is the hardware key. Bypass the algorithm by connecting directly with verified alumni and accessing data straight from the source.
        </p>

        {/* Hardware-Style Buttons */}
        <div className="flex flex-col sm:flex-row gap-6">
          <Link to="/forum" className="group relative px-8 py-4 bg-zinc-900 text-white font-mono text-sm font-bold uppercase tracking-widest rounded-xl shadow-[0_6px_0_#3f3f46] hover:translate-y-[2px] hover:shadow-[0_4px_0_#3f3f46] active:translate-y-[6px] active:shadow-none transition-all flex items-center gap-3">
            <Terminal className="w-4 h-4 text-orange-500" /> Forums
          </Link>
          <Link to="/match" className="group relative px-8 py-4 bg-white text-zinc-900 border-2 border-zinc-200 font-mono text-sm font-bold uppercase tracking-widest rounded-xl shadow-[0_6px_0_#e4e4e7] hover:translate-y-[2px] hover:shadow-[0_4px_0_#e4e4e7] active:translate-y-[6px] active:shadow-none transition-all flex items-center gap-3">
            <Activity className="w-4 h-4 text-zinc-400" />  Matchmaking
          </Link>
        </div>
      </div>

      {/* 2. DIAGNOSTICS / THE "WHY" SECTION */}
      <div className="bg-white border-t-2 border-zinc-100 py-24 relative z-10">
        <div className="max-w-6xl mx-auto px-4">
          
          {/* Section Header */}
          <div className="flex items-center gap-4 mb-16">
            <div className="h-px bg-zinc-200 flex-1"></div>
            <h2 className="font-mono text-xs font-bold tracking-widest text-zinc-400 uppercase flex items-center gap-2">
              <BarChart3 className="w-4 h-4" /> System Diagnostics: The Job Market
            </h2>
            <div className="h-px bg-zinc-200 flex-1"></div>
          </div>

          {/* Data Cards Grid */}
          <div className="grid lg:grid-cols-3 gap-8 mb-24">
            
            {/* Card 1: Equalizer Bar Graph (The Void) */}
            <div className="bg-white border-2 border-zinc-200 p-8 rounded-3xl shadow-sm hover:border-orange-200 transition-colors group">
              <div className="font-mono text-[10px] text-zinc-400 mb-6 uppercase tracking-wider flex justify-between">
                <span>Cold App Yield</span>
                <span className="text-zinc-300">SIG_01</span>
              </div>
              
              {/* Synthesizer Equalizer Chart */}
              <div className="flex items-end justify-between h-32 mb-6 border-b-2 border-zinc-900 pb-2 relative">
                <div className="absolute top-1/4 left-0 w-full border-t border-dashed border-zinc-200"></div>
                <div className="absolute top-2/4 left-0 w-full border-t border-dashed border-zinc-200"></div>
                <div className="absolute top-3/4 left-0 w-full border-t border-dashed border-zinc-200"></div>
                
                <div className="w-[15%] bg-zinc-200 h-[90%] rounded-sm relative z-10 hover:bg-zinc-300 transition-colors"></div>
                <div className="w-[15%] bg-zinc-200 h-[65%] rounded-sm relative z-10 hover:bg-zinc-300 transition-colors"></div>
                <div className="w-[15%] bg-zinc-200 h-[40%] rounded-sm relative z-10 hover:bg-zinc-300 transition-colors"></div>
                <div className="w-[15%] bg-zinc-200 h-[15%] rounded-sm relative z-10 hover:bg-zinc-300 transition-colors"></div>
                <div className="w-[15%] bg-[#7AAACE] h-[2.4%] rounded-sm shadow-[0_0_15px_rgba(249,115,22,0.6)] relative z-10">
                  <div className="absolute -top-8 left-1/2 -translate-x-1/2 font-mono text-[10px] text-[#7AAACE] font-bold bg-orange-50 px-2 py-1 rounded">2.4%</div>
                </div>
              </div>
              
              <div className="text-5xl font-black text-zinc-900 mb-2 tracking-tighter">2.4%</div>
              <p className="text-sm text-zinc-500 font-medium leading-relaxed">The average interview rate when applying cold. Without a referral, your resume goes straight to the void.</p>
            </div>

            {/* Card 2: Oscilloscope Line Graph (Time to Hire) */}
            <div className="bg-zinc-900 border-2 border-zinc-800 p-8 rounded-3xl shadow-lg text-white group relative overflow-hidden">
              {/* Faint grid background for the terminal look */}
              <div className="absolute inset-0 bg-[linear-gradient(to_right,#27272a_1px,transparent_1px),linear-gradient(to_bottom,#27272a_1px,transparent_1px)] bg-[size:1rem_1rem] opacity-30"></div>

              <div className="font-mono text-[10px] text-zinc-500 mb-6 uppercase tracking-wider flex justify-between relative z-10">
                <span>Time to Hire</span>
                <span className="text-[#7AAACE] animate-pulse">DELAY</span>
              </div>
              
              {/* Oscilloscope SVG Chart */}
              <div className="relative h-32 mb-6 border-b-2 border-zinc-700 pb-2 z-10">
                <svg className="w-full h-full overflow-visible" viewBox="0 0 100 40" preserveAspectRatio="none">
                  {/* Grid Lines inside SVG */}
                  <line x1="0" y1="10" x2="100" y2="10" stroke="#3f3f46" strokeWidth="0.5" strokeDasharray="2 2" />
                  <line x1="0" y1="20" x2="100" y2="20" stroke="#3f3f46" strokeWidth="0.5" strokeDasharray="2 2" />
                  <line x1="0" y1="30" x2="100" y2="30" stroke="#3f3f46" strokeWidth="0.5" strokeDasharray="2 2" />
                  
                  {/* Curved Path for a more analog hardware feel */}
                  <path d="M0,35 Q30,35 50,25 T100,5" fill="none" stroke="#f97316" strokeWidth="2.5" className="drop-shadow-[0_0_8px_rgba(249,115,22,0.8)]"/>
                  
                  {/* Data Nodes */}
                  <circle cx="0" cy="35" r="2" fill="#fff" stroke="#f97316" strokeWidth="1"/>
                  <circle cx="50" cy="25" r="2" fill="#fff" stroke="#f97316" strokeWidth="1"/>
                  <circle cx="100" cy="5" r="3" fill="#f97316" className="animate-ping" style={{ transformOrigin: '100px 5px' }}/>
                  <circle cx="100" cy="5" r="2" fill="#fff"/>
                </svg>
              </div>
              
              <div className="text-5xl font-black text-white mb-2 tracking-tighter relative z-10">68 <span className="text-xl text-zinc-500 tracking-normal font-bold">days</span></div>
              <p className="text-sm text-zinc-400 font-medium leading-relaxed relative z-10">The hiring process is freezing up globally. Bypassing the HR queue is no longer optional; it is required.</p>
            </div>

            {/* Card 3: Node Matrix (The Multiplier) */}
            <div className="bg-white border-2 border-zinc-200 p-8 rounded-3xl shadow-sm flex flex-col justify-between hover:border-orange-200 transition-colors">
              <div>
                <div className="font-mono text-[10px] text-zinc-400 mb-6 uppercase tracking-wider flex justify-between">
                  <span>The Multiplier</span>
                  <span className="text-zinc-300">NET_14</span>
                </div>
                
                {/* Visual Node Matrix */}
                <div className="h-32 mb-6 flex items-center justify-center border-b-2 border-zinc-100 pb-2">
                  <div className="relative w-32 h-32">
                    {/* Center Node */}
                    <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-4 h-4 bg-orange-500 rounded-full shadow-[0_0_15px_rgba(249,115,22,0.6)] z-10"></div>
                    {/* Connecting Lines & Outer Nodes */}
                    {[0, 45, 90, 135, 180, 225, 270, 315].map((deg, i) => (
                      <div key={i} className="absolute top-1/2 left-1/2 w-full h-[1px] bg-zinc-200 origin-left" style={{ transform: `translateY(-50%) rotate(${deg}deg)` }}>
                        <div className="absolute right-0 top-1/2 -translate-y-1/2 w-2 h-2 bg-zinc-800 rounded-full"></div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="text-5xl font-black text-zinc-900 mb-2 tracking-tighter">14x</div>
                <p className="text-sm text-zinc-500 font-medium leading-relaxed">Applicants with direct alumni referrals are 14 times more likely to receive an offer compared to the standard pipeline.</p>
              </div>
            </div>
          </div>

          {/* 3. INTERACTIVE STATS ENGINE */}
          <div className="max-w-5xl mx-auto bg-white rounded-3xl shadow-xl border-2 border-zinc-200 overflow-hidden flex flex-col md:flex-row mt-12 mb-12">
            
            {/* Left Side: The Pitch */}
            <div className="md:w-5/12 p-10 md:p-12 flex flex-col justify-center bg-[#F7F8F0]">
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-[#9CD5FF]/30 text-[#355872] border border-[#7AAACE]/30 font-bold text-[10px] uppercase tracking-widest w-fit mb-6">
                <Target className="w-3 h-3" /> Trajectory Engine
              </div>
              <h2 className="text-3xl font-black text-zinc-900 leading-tight mb-4 uppercase tracking-tighter">
                Calibrate Your <span className="text-[#355872]">Market Value</span>
              </h2>
              <p className="text-zinc-600 font-medium text-sm leading-relaxed mb-8">
                Input your current academic standing. Our proprietary engine will cross-reference our verified database to project your expected compensation and immediately identify alumni who can get you there.
              </p>
              <div className="flex items-center gap-4 text-xs font-mono font-bold text-zinc-400 uppercase tracking-widest">
                <Database className="w-4 h-4 text-[#7AAACE]" /> 100+ Verified Data Points
              </div>
            </div>

            {/* Right Side: The Form */}
            <div className="md:w-7/12 bg-zinc-900 p-10 md:p-12 text-white relative">
              {/* Decorative Grid */}
              <div className="absolute inset-0 bg-[linear-gradient(to_right,#3f3f46_1px,transparent_1px),linear-gradient(to_bottom,#3f3f46_1px,transparent_1px)] bg-[size:1rem_1rem] opacity-10"></div>
              
              <div className="relative z-10 space-y-6">
                
                <div className="grid grid-cols-2 gap-6">
                  <div className="space-y-2">
                    <label className="text-[10px] font-mono text-zinc-400 uppercase tracking-widest">Target University</label>
                    <select 
                      value={school} 
                      onChange={(e) => setSchool(e.target.value)}
                      className="w-full bg-zinc-800 border-2 border-zinc-700 text-white text-sm rounded-xl p-3 outline-none focus:border-[#7AAACE] transition-colors font-medium"
                    >
                      {/* You can replace this array with {SCHOOLS.map(...)} if you imported them */}
                      {["Wharton", "Harvard", "Stanford", "NYU Stern", "UChicago", "MIT", "Columbia"].map(s => <option key={s} value={s}>{s}</option>)}
                    </select>
                  </div>
                  <div className="space-y-2">
                    <label className="text-[10px] font-mono text-zinc-400 uppercase tracking-widest">Current Major</label>
                    <select 
                      value={major} 
                      onChange={(e) => setMajor(e.target.value)}
                      className="w-full bg-zinc-800 border-2 border-zinc-700 text-white text-sm rounded-xl p-3 outline-none focus:border-[#7AAACE] transition-colors font-medium"
                    >
                      {/* You can replace this array with {MAJORS.map(...)} if you imported them */}
                      {["Finance", "Economics", "Computer Science", "Mathematics", "Data Science", "Business"].map(m => <option key={m} value={m}>{m}</option>)}
                    </select>
                  </div>
                </div>

                <div className="space-y-2">
                  <label className="text-[10px] font-mono text-zinc-400 uppercase tracking-widest">Cumulative GPA</label>
                  <input 
                    type="number" 
                    step="0.1"
                    min="0"
                    max="4.0"
                    value={gpa}
                    onChange={(e) => setGpa(e.target.value)}
                    className="w-full bg-zinc-800 border-2 border-zinc-700 text-white text-sm rounded-xl p-3 outline-none focus:border-[#7AAACE] transition-colors font-medium"
                  />
                </div>

                <div className="pt-6 border-t-2 border-zinc-800 grid sm:grid-cols-2 gap-4">
                  <button 
                    onClick={() => handleAnalyze('/salaries')}
                    disabled={loadingPath !== null}
                    className="w-full bg-[#355872] hover:bg-[#2c4b61] text-white py-4 rounded-xl font-black text-xs uppercase tracking-widest transition-all shadow-[0_4px_0_#1a2d3a] hover:translate-y-[2px] hover:shadow-[0_2px_0_#1a2d3a] active:translate-y-[4px] active:shadow-none flex items-center justify-center gap-2"
                  >
                    {loadingPath === '/salaries' ? <Loader2 className="w-4 h-4 animate-spin" /> : <DollarSign className="w-4 h-4" />}
                    {loadingPath === '/salaries' ? 'Crunching...' : 'View Salaries'}
                  </button>
                  
                  <button 
                    onClick={() => handleAnalyze('/match')}
                    disabled={loadingPath !== null}
                    className="w-full bg-zinc-800 hover:bg-zinc-700 border-2 border-zinc-600 hover:border-orange-500/50 text-white py-4 rounded-xl font-black text-xs uppercase tracking-widest transition-all flex items-center justify-center gap-2 group"
                  >
                    {loadingPath === '/match' ? <Loader2 className="w-4 h-4 animate-spin text-orange-500" /> : <Zap className="w-4 h-4 text-orange-500 group-hover:scale-110 transition-transform" />}
                    {loadingPath === '/match' ? 'Scanning...' : 'Find Matches'}
                  </button>
                </div>

              </div>
            </div>
          </div>

        </div>
      </div>
    </div>
  );
};

// --- HOME / LANDING PAGE ---
const Home = () => {
  const navigate = useNavigate();
  const [loadingPath, setLoadingPath] = useState<string | null>(null);

  // Form State
  const [school, setSchool] = useState(SCHOOLS[0]);
  const [major, setMajor] = useState(MAJORS[0]);
  const [gpa, setGpa] = useState("3.8");

  const handleAnalyze = (path: string) => {
    setLoadingPath(path);
    // Simulate the "Engine Crunching Data" for 1.2 seconds before routing
    setTimeout(() => {
      navigate(path);
    }, 1200);
  };

  return (
    <div className="animate-in fade-in duration-700 space-y-20 pb-20">
      
      {/* 1. YOUR EXISTING GRAPH / HERO SECTION */}
      {/* (I am using a dark placeholder here, but you can drop your actual Diagnostics graph or hero text here) */}
      <div className="h-[60vh] min-h-[400px] bg-zinc-900 rounded-3xl border border-zinc-800 flex items-center justify-center shadow-2xl relative overflow-hidden">
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#27272a_1px,transparent_1px),linear-gradient(to_bottom,#27272a_1px,transparent_1px)] bg-[size:2rem_2rem] opacity-30"></div>
        <div className="relative z-10 text-center space-y-4">
          <Activity className="w-16 h-16 text-[#7AAACE] mx-auto animate-pulse" />
          <h1 className="text-4xl md:text-6xl font-black text-white tracking-tighter uppercase italic">
            Qualifly <span className="text-[#7AAACE]">Terminal</span>
          </h1>
          <p className="text-zinc-400 font-medium tracking-widest uppercase text-sm">Scroll down to initialize your trajectory</p>
        </div>
      </div>

      {/* 2. THE STATS INPUT ENGINE */}
      <div className="max-w-5xl mx-auto bg-white rounded-3xl shadow-xl border overflow-hidden flex flex-col md:flex-row">
        
        {/* Left Side: The Pitch */}
        <div className="md:w-5/12 p-10 md:p-12 flex flex-col justify-center bg-[#F7F8F0]">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-100 text-blue-700 font-bold text-[10px] uppercase tracking-widest w-fit mb-6">
            <Target className="w-3 h-3" /> Trajectory Engine
          </div>
          <h2 className="text-3xl font-black text-gray-900 leading-tight mb-4 uppercase tracking-tighter">
            Calibrate Your <span className="text-[#355872]">Market Value</span>
          </h2>
          <p className="text-gray-600 font-medium text-sm leading-relaxed mb-8">
            Input your current academic standing. Our proprietary engine will cross-reference our verified database to project your expected compensation and immediately identify alumni who can get you there.
          </p>
          <div className="flex items-center gap-4 text-xs font-bold text-gray-400 uppercase tracking-widest">
            <Database className="w-4 h-4 text-[#7AAACE]" /> {MOCK_SALARIES.length + MOCK_MATCHES.length} Verified Data Points
          </div>
        </div>

        {/* Right Side: The Form */}
        <div className="md:w-7/12 bg-zinc-900 p-10 md:p-12 text-white relative">
          {/* Decorative Grid */}
          <div className="absolute inset-0 bg-[linear-gradient(to_right,#3f3f46_1px,transparent_1px),linear-gradient(to_bottom,#3f3f46_1px,transparent_1px)] bg-[size:1rem_1rem] opacity-10"></div>
          
          <div className="relative z-10 space-y-6">
            
            <div className="grid grid-cols-2 gap-6">
              <div className="space-y-2">
                <label className="text-[10px] font-mono text-zinc-400 uppercase tracking-widest">Target University</label>
                <select 
                  value={school} 
                  onChange={(e) => setSchool(e.target.value)}
                  className="w-full bg-zinc-800 border border-zinc-700 text-white text-sm rounded-xl p-3 outline-none focus:border-[#7AAACE] transition-colors appearance-none font-medium"
                >
                  {SCHOOLS.map(s => <option key={s} value={s}>{s}</option>)}
                </select>
              </div>
              <div className="space-y-2">
                <label className="text-[10px] font-mono text-zinc-400 uppercase tracking-widest">Current Major</label>
                <select 
                  value={major} 
                  onChange={(e) => setMajor(e.target.value)}
                  className="w-full bg-zinc-800 border border-zinc-700 text-white text-sm rounded-xl p-3 outline-none focus:border-[#7AAACE] transition-colors appearance-none font-medium"
                >
                  {MAJORS.map(m => <option key={m} value={m}>{m}</option>)}
                </select>
              </div>
            </div>

            <div className="space-y-2">
              <label className="text-[10px] font-mono text-zinc-400 uppercase tracking-widest">Cumulative GPA</label>
              <input 
                type="number" 
                step="0.1"
                min="0"
                max="4.0"
                value={gpa}
                onChange={(e) => setGpa(e.target.value)}
                className="w-full bg-zinc-800 border border-zinc-700 text-white text-sm rounded-xl p-3 outline-none focus:border-[#7AAACE] transition-colors font-medium"
              />
            </div>

            <div className="pt-6 border-t border-zinc-800 grid sm:grid-cols-2 gap-4">
              <button 
                onClick={() => handleAnalyze('/salaries')}
                disabled={loadingPath !== null}
                className="w-full bg-[#355872] hover:bg-[#2c4b61] text-white py-4 rounded-xl font-black text-xs uppercase tracking-widest transition-all shadow-[0_0_20px_rgba(53,88,114,0.4)] flex items-center justify-center gap-2"
              >
                {loadingPath === '/salaries' ? <Loader2 className="w-4 h-4 animate-spin" /> : <DollarSign className="w-4 h-4" />}
                {loadingPath === '/salaries' ? 'Crunching...' : 'View Salaries'}
              </button>
              
              <button 
                onClick={() => handleAnalyze('/match')}
                disabled={loadingPath !== null}
                className="w-full bg-zinc-800 hover:bg-zinc-700 border border-zinc-600 hover:border-amber-500/50 text-white py-4 rounded-xl font-black text-xs uppercase tracking-widest transition-all flex items-center justify-center gap-2 group"
              >
                {loadingPath === '/match' ? <Loader2 className="w-4 h-4 animate-spin text-amber-500" /> : <Zap className="w-4 h-4 text-amber-500 group-hover:scale-110 transition-transform" />}
                {loadingPath === '/match' ? 'Scanning...' : 'Find Matches'}
              </button>
            </div>
          
          </div>
        </div>
      </div>
    </div>
  );
};

export default function App() {
  return (
    <AppProvider>
      <BrowserRouter>
        <Layout>
          <Routes>
            <Route path="/" element={<Landing />} /> 
            <Route path="/jobs" element={<JobsExplorer></JobsExplorer>} />
            <Route path="/forum" element={<Forum />} />
            <Route path="/forum/:id" element={<ThreadDetail />} />
            <Route path="/salaries" element={<SalaryExplorer />} />
            <Route path="/match" element={<Matchmaking />} />
            <Route path="/vault" element={<ResumeVault />} />
            <Route path="/diagnostics" element={<Diagnostics />} />
          </Routes>
        </Layout>
      </BrowserRouter>
    </AppProvider>
  );
}