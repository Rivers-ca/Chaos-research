export type Job = {
  id: string;
  title: string;
  company: string;
  location: string;
  isHot: boolean;
};
export type Match = { id: string; name: string; role: string; company: string; school: string; club: string; score: number; };
export type Resume = {
  id: string;
  role: string;
  firm: string;
  school: string;
  gpa: string;
  year: string;
};
export type Comment = {
  id: string;
  author: string;
  text: string;
  isAlumni: boolean;
};

export type Thread = {
  id: string;
  tier: string;
  school: string;
  title: string;
  body: string;
  author: string;
  isAlumni: boolean;
  comments: Comment[];
};
export type Salary = { id: string; role: string; school: string; major: string; lowPay: number; medianPay: number; highPay: number; outcomes: string[]; };

export const SCHOOLS = [
  "Wharton",
  "Harvard",
  "Stanford",
  "NYU Stern",
  "UChicago",
  "MIT",
  "Columbia"
];

export const MAJORS = [
  "Finance",
  "Economics",
  "Computer Science",
  "Mathematics",
  "Data Science",
  "Business"
];

export const CLUBS = [
  "Investment Banking Club",
  "Consulting Group",
  "PE/VC Network",
  "Women in Business",
  "Quant Society"
];

export const MOCK_JOBS: Job[] = [
  { id: "1", title: "2025 Summer Analyst - Investment Banking", company: "Goldman Sachs", location: "New York, NY", isHot: true },
  { id: "2", title: "Associate Consultant", company: "Bain & Company", location: "Boston, MA", isHot: true },
  { id: "3", title: "Software Engineer (New Grad)", company: "Google", location: "Mountain View, CA", isHot: false },
  { id: "4", title: "Private Equity Associate", company: "Blackstone", location: "New York, NY", isHot: true },
  { id: "5", title: "Quantitative Researcher", company: "Jane Street", location: "New York, NY", isHot: true },
  { id: "6", title: "Product Manager (APM)", company: "Meta", location: "Menlo Park, CA", isHot: false },
  { id: "7", title: "Investment Banking Analyst", company: "Morgan Stanley", location: "San Francisco, CA", isHot: false },
  { id: "8", title: "Strategy Consultant", company: "McKinsey & Company", location: "Chicago, IL", isHot: true },
  { id: "9", title: "Venture Capital Analyst", company: "Sequoia Capital", location: "Menlo Park, CA", isHot: true },
  { id: "10", title: "Data Scientist", company: "Stripe", location: "Remote", isHot: false },
  { id: "11", title: "Restructuring Analyst", company: "Evercore", location: "New York, NY", isHot: false },
  { id: "12", title: "Global Markets Summer Analyst", company: "J.P. Morgan", location: "London, UK", isHot: false },
];

export const MOCK_THREADS: Thread[] = [
  {
    id: "1",
    title: "Megafund PE recruiting timeline moved up AGAIN?",
    body: "Discussion about accelerated private equity timelines and how it affects junior bankers.",
    author: "capital_gains",
    tier: "Alumni Insights",
    school: "Wharton",
    isAlumni: true,
    comments: [
      { id: "c1", author: "WhartIB", text: "This cycle is insane. A lot of people I know are good have been rejected, you mightt need to take a year or two to get an MBA.", isAlumni: true },
      { id: "c2", author: "PEfromIB", text: "Agree, timelines are compressed. You need to really focus on your resume, try some of the samples, mines there.", isAlumni: false },
    ]
  },
  {
    id: "2",
    title: "Roast my resume - Target school, low GPA",
    body: "Looking for feedback on how to offset GPA with deal experience.",
    author: "stressed_junior",
    tier: "Student Hub",
    school: "NYU Stern",
    isAlumni: false,
    comments: [
      { id: "c3", author: "MistaPE", text: "Yes, take your time on a MBA, you need a solid base to even try to break in now and days.", isAlumni: false },
    ]
  },
  {
    id: "3",
    title: "McKinsey vs. Bain for Tech Exits?",
    body: "Which firm places better into product and VC?",
    author: "strategy_guru",
    tier: "Job Discussions",
    school: "Harvard",
    isAlumni: true,
    comments: []
  },
  {
    id: "4",
    title: "Whats the best consulting club at Columbia?",
    body: "I am freshmen and I am looking to join a consulting club. Which one is the best and why?",
    author: "consulting_freshman",
    tier: "Alumni Insights",
    school: "Columbia",
    isAlumni: false,
    comments: [
      { id: "c1", author: "Mckinseyman", text: "I would think the best bet for someone with in your position is to try to land a internship instead of a club.", isAlumni: true },
      { id: "c2", author: "Delloite", text: "I agree with McKinseyMan 100%, going to Columbia already puts you ahead of the game. I landed my first internship thruogh cold ccalling and interview prep.", isAlumni: false }
    ]
  },
  {
    id: "5",
    title: "Jane Street vs Citadel Culture?",
    body: "Which one has a better work-life balance for Quants?",
    author: "math_wiz",
    tier: "Job Discussions",
    school: "MIT",
    isAlumni: false,
    comments: [
      { id: "c1", author: "MitExpert", text: "I did not do clubs at all and I was able to break in, maybe look at some other resumes to see what people even talk about.", isAlumni: true },
      { id: "c2", author: "Throwback_dino", text: "I agree with MitExpert 100%, going to MIT already puts you ahead of the game. I landed my first internship thruogh cold ccalling and interview prep.", isAlumni: true },
      { id: "c3", author: "Mckinseyman", text: "I would think the best bet for someone with in your position is to try to land a internship instead of a club, .", isAlumni: false },
      { id: "c4", author: "Delloite", text: "I agree with McKinseyMan 100%, going to MitExpert already puts you ahead of the game. I landed my first internship thruogh cold ccalling and interview prep.", isAlumni: false }
  
    ]
  },
  {
    id: "6",
    title: "Are Clubs Needed?",
    body: "I want to join a club but I am not sure if it is worth the time investment. Do you think clubs are necessary for recruiting?",
    author: "quant_enthusiast",
    tier: "Student Hub",
    school: "MIT",
    isAlumni: false,
    comments: [
      { id: "c1", author: "MitExpert", text: "I did not do clubs at all and I was able to break in, maybe look at some other resumes to see what people even talk about.", isAlumni: true },
      { id: "c2", author: "Throwback_dino", text: "I agree with MitExpert 100%, going to MIT already puts you ahead of the game. I landed my first internship thruogh cold ccalling and interview prep.", isAlumni: true },
      { id: "c3", author: "Mckinseyman", text: "I would think the best bet for someone with in your position is to try to land a internship instead of a club, .", isAlumni: false },
      { id: "c4", author: "Delloite", text: "I agree with McKinseyMan 100%, going to MitExpert already puts you ahead of the game. I landed my first internship thruogh cold ccalling and interview prep.", isAlumni: false }
    ]
  }
];

export const MOCK_MATCHES: Match[] = [
  { id: "1", name: "Sarah Jenkins", role: "Associate", company: "McKinsey", school: "Harvard", club: "Consulting Group", score: 98 },
  { id: "2", name: "David Chen", role: "Analyst", company: "Goldman Sachs", school: "Wharton", club: "Investment Banking Club", score: 95 },
  { id: "3", name: "Elena Rodriguez", role: "Product Manager", company: "Google", school: "Stanford", club: "Women in Business", score: 92 },
  { id: "4", name: "Michael Chang", role: "Quant Researcher", company: "Citadel", school: "MIT", club: "Quant Society", score: 89 },
  { id: "5", name: "Jessica Taylor", role: "PE Associate", company: "KKR", school: "NYU Stern", club: "PE/VC Network", score: 88 },
  { id: "6", name: "James Wilson", role: "Strategy", company: "Uber", school: "UChicago", club: "Consulting Group", score: 85 },
  { id: "7", name: "Chloe Kim", role: "IB Analyst", company: "Evercore", school: "Columbia", club: "Investment Banking Club", score: 82 },
  { id: "8", name: "Marcus Johnson", role: "VC Associate", company: "Andreessen Horowitz", school: "Stanford", club: "PE/VC Network", score: 80 },
];

export const MOCK_RESUMES: Resume[] = [
  { id: "1", role: "Investment Banking Analyst", firm: "Goldman Sachs", school: "Wharton", gpa: "3.92", year: "2024" },
  { id: "2", role: "Associate Consultant", firm: "Bain & Company", school: "Harvard", gpa: "3.85", year: "2023" },
  { id: "3", role: "Software Engineer", firm: "Google", school: "Stanford", gpa: "3.78", year: "2024" },
  { id: "4", role: "Private Equity Associate", firm: "Blackstone", school: "NYU Stern", gpa: "3.95", year: "2022" },
  { id: "5", role: "Quantitative Researcher", firm: "Jane Street", school: "MIT", gpa: "4.00", year: "2023" },
  { id: "6", role: "Product Manager", firm: "Meta", school: "UChicago", gpa: "3.81", year: "2024" },
  { id: "7", role: "Restructuring Analyst", firm: "PJT Partners", school: "Columbia", gpa: "3.88", year: "2023" },
  { id: "8", role: "Venture Capital Analyst", firm: "Sequoia Capital", school: "Stanford", gpa: "3.91", year: "2022" },
];

export const MOCK_SALARIES = [
  { id: 1, role: "Investment Banking Analyst", major: "Finance", school: "Wharton", medianPay: 175000, lowPay: 140000, highPay: 210000, outcomes: ["Goldman Sachs", "Morgan Stanley", "Evercore"] },
  { id: 2, role: "Associate Consultant", major: "Economics", school: "Harvard", medianPay: 115000, lowPay: 100000, highPay: 130000, outcomes: ["McKinsey", "Bain", "BCG"] },
  { id: 3, role: "Software Engineer (L3)", major: "Computer Science", school: "Stanford", medianPay: 185000, lowPay: 150000, highPay: 220000, outcomes: ["Google", "Meta", "Stripe"] },
  { id: 4, role: "Quantitative Researcher", major: "Mathematics", school: "MIT", medianPay: 350000, lowPay: 250000, highPay: 450000, outcomes: ["Jane Street", "Citadel", "Two Sigma"] },
  { id: 5, role: "Private Equity Associate", major: "Finance", school: "NYU Stern", medianPay: 275000, lowPay: 220000, highPay: 320000, outcomes: ["Blackstone", "KKR", "Apollo"] },
  { id: 6, role: "Product Manager (APM)", major: "Computer Science", school: "UChicago", medianPay: 155000, lowPay: 130000, highPay: 180000, outcomes: ["Google", "Uber", "Airbnb"] },
  { id: 7, role: "Strategy Analyst", major: "Business", school: "Columbia", medianPay: 105000, lowPay: 90000, highPay: 120000, outcomes: ["Deloitte", "Accenture", "Oliver Wyman"] }
];