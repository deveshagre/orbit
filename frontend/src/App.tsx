import { FormEvent, useEffect, useRef, useState } from 'react'
import { ArrowUp, Check, CircleAlert, Globe, LoaderCircle, Menu, Plus, Sparkles, Wrench, X } from 'lucide-react'
import orbitLogo from '../logo.png'

type Activity = { tool: string; ok: boolean; summary: string }
type Timing = { mode: 'simple' | 'tools'; total_ms: number; model_ms: number }
type Message = { role: 'user' | 'assistant'; content: string; activity?: Activity[]; timing?: Timing; error?: boolean }
type Conversation = { id: string; title: string; messages: Message[] }
const welcome = 'What would you like to explore today?'

// Base API URL configurable via VITE_API_URL (empty string uses Vite dev proxy)
const API_BASE = (import.meta.env.VITE_API_URL || '').replace(/\/+$/, '')

export default function App() {
  const [conversations, setConversations] = useState<Conversation[]>([{id: crypto.randomUUID(), title: 'New conversation', messages: []}])
  const [activeId, setActiveId] = useState(conversations[0].id)
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [menuOpen, setMenuOpen] = useState(false)
  const [health, setHealth] = useState<'checking'|'ready'|'offline'>('checking')
  const [modelName, setModelName] = useState('Gemini')
  const bottom = useRef<HTMLDivElement>(null)
  const active = conversations.find(c => c.id === activeId) ?? conversations[0]

  useEffect(() => {
    fetch(`${API_BASE}/api/health`)
      .then(async r => {
        if (!r.ok) {
          setHealth('offline')
          return
        }
        const data = await r.json().catch(() => ({}))
        if (data.ok) {
          setHealth('ready')
          if (data.model) {
            setModelName(data.model)
          }
        } else {
          setHealth('offline')
        }
      })
      .catch(() => setHealth('offline'))
  }, [])

  useEffect(() => { bottom.current?.scrollIntoView({behavior:'smooth'}) }, [active?.messages, busy])
  const update = (id: string, fn: (c: Conversation) => Conversation) => setConversations(cs => cs.map(c => c.id === id ? fn(c) : c))
  const fresh = () => { const id = crypto.randomUUID(); setConversations(cs => [{id, title:'New conversation', messages:[]}, ...cs]); setActiveId(id); setInput(''); setMenuOpen(false) }

  async function send(e?: FormEvent, suggestion?: string) {
    e?.preventDefault(); const text = (suggestion ?? input).trim(); if (!text || busy) return
    const id = active.id
    update(id, c => ({...c, title: c.messages.length ? c.title : text.slice(0,34), messages:[...c.messages, {role:'user',content:text}, {role:'assistant',content:'',activity:[]}]}))
    setInput(''); setBusy(true)
    let streamedContent = ''
    const patchAssistant = (patch: Partial<Message>) => update(id, c => {
      const messages = [...c.messages]
      const index = messages.length - 1
      if (index >= 0 && messages[index].role === 'assistant') messages[index] = {...messages[index], ...patch}
      return {...c, messages}
    })

    try {
      const res = await fetch(`${API_BASE}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, conversation_id: id })
      })

      if (!res.ok || !res.body) {
        const data = await res.json().catch(() => ({}))
        throw new Error(data.error || `Request failed with status ${res.status}`)
      }

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      const handle = (raw: string) => {
        const line = raw.split('\n').find(part => part.startsWith('data: '))
        if (!line) return
        const event = JSON.parse(line.slice(6))
        if (event.type === 'delta') { streamedContent += event.text; patchAssistant({content:streamedContent}) }
        if (event.type === 'activity') update(id, c => {
          const messages = [...c.messages]
          const index = messages.length - 1
          if (index >= 0 && messages[index].role === 'assistant') {
            const activities = [...(messages[index].activity || [])]
            const itemIndex = activities.findIndex(a => a.tool === event.activity.tool)
            if (itemIndex >= 0) activities[itemIndex] = event.activity
            else activities.push(event.activity)
            messages[index] = {...messages[index], activity:activities}
          }
          return {...c, messages}
        })
        if (event.type === 'done') patchAssistant({content:event.answer || streamedContent, activity:event.tool_activity || [], timing:event.timing})
        if (event.type === 'error') patchAssistant({content:event.error || 'ORBIT could not complete the request.', error:true})
      }

      while (true) {
        const {value, done} = await reader.read()
        buffer += decoder.decode(value || new Uint8Array(), {stream:!done})
        const parts = buffer.split('\n\n')
        buffer = parts.pop() || ''
        parts.forEach(handle)
        if (done) break
      }
      if (buffer.trim()) handle(buffer)
    } catch (err) {
      patchAssistant({content:err instanceof Error ? err.message : 'Could not connect to ORBIT.', error:true})
    } finally {
      setBusy(false)
    }
  }

  return <div className="app-shell min-h-screen">
    <aside className={`sidebar ${menuOpen?'sidebar-open':''}`}>
      <div className="brand-row">
        <img className="sidebar-logo" src={orbitLogo} alt="ORBIT"/>
        <button className="icon-btn mobile-close" onClick={()=>setMenuOpen(false)} aria-label="Close sidebar"><X size={18}/></button>
      </div>
      <button className="new-chat" onClick={fresh}><Plus size={17}/><span>New chat</span><kbd>⌘ K</kbd></button>
      <div className="side-label">RECENTS</div>
      <div className="conversation-list">
        {conversations.map(c=><button key={c.id} onClick={()=>{setActiveId(c.id);setMenuOpen(false)}} className={`conversation-link ${c.id===activeId?'selected':''}`}><span className="conv-dot"/>{c.title}</button>)}
      </div>
      <div className="sidebar-bottom">
        <div className={`connection ${health}`}><span/>{health==='ready'?'API ready':health==='offline'?'API unavailable':'Connecting'}</div>
        <div className="local-note">
          <span className="local-icon"><Sparkles size={15}/></span>
          <div><strong>Cloud AI Agent</strong><small>Powered by Gemini API</small></div>
        </div>
      </div>
    </aside>
    {menuOpen && <button className="scrim" onClick={()=>setMenuOpen(false)} aria-label="Close menu"/>}
    <main className="main-panel">
      <header className="topbar">
        <button className="icon-btn menu-trigger" onClick={()=>setMenuOpen(true)} aria-label="Open sidebar"><Menu size={19}/></button>
        <div className="top-title">ORBIT <span className="top-separator">/</span> <span>Cloud agent</span></div>
        <div className="model-pill"><i/> {modelName}</div>
      </header>
      <section className={`chat-area ${active.messages.length?'has-messages':''}`}>
        {active.messages.length===0 ? (
          <div className="welcome">
            <img className="welcome-logo" src={orbitLogo} alt="ORBIT" />
            <div className="eyebrow">CLOUD PERSONAL AGENT</div>
            <h1>{welcome}</h1>
            <p>Search the live web and solve complex problems.<br/>Powered by Google Gemini in the cloud.</p>
            <div className="suggestions">
              {[
                'Search the web for the latest Python release',
                'Explain how cloud deployment works',
                'Compare React with Next.js for web apps'
              ].map((s,i)=>(
                <button onClick={()=>send(undefined,s)} key={s}>
                  <span>{i===0?<Globe size={14}/>:i===1?'▤':'✳'}</span>
                  {s}
                  <ArrowUp size={14}/>
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="messages">
            {active.messages.map((m,i)=>(
              <div className={`message-row ${m.role}`} key={i}>
                <div className={`avatar ${m.role}`}>
                  {m.role==='assistant'?<Sparkles size={15}/>:<span>Y</span>}
                </div>
                <div className="message-content">
                  {m.role==='assistant'&&m.activity?.length ? (
                    <div className="activity-list">
                      {m.activity.map((a,j)=>(
                        <div className={`activity ${a.ok?'':'failed'}`} key={j}>
                          <span className="activity-icon">{a.ok?<Check size={12}/>:<CircleAlert size={12}/>}</span>
                          <Wrench size={13}/>
                          <span>{a.tool.replace(/_/g,' ')}</span>
                          <span className="activity-summary">{a.summary}</span>
                        </div>
                      ))}
                    </div>
                  ):null}
                  {m.content ? (
                    <div className={`bubble ${m.role} ${m.error?'error':''}`}>{m.content}</div>
                  ) : m.role==='assistant'&&busy ? (
                    <div className="thinking"><LoaderCircle size={15} className="spin"/><span>ORBIT is thinking</span></div>
                  ) : null}
                  {m.role==='assistant'&&m.timing ? (
                    <div className="message-latency">{(m.timing.total_ms/1000).toFixed(1)}s · {m.timing.mode==='simple'?'direct response':'tools'}</div>
                  ) : null}
                </div>
              </div>
            ))}
            <div ref={bottom}/>
          </div>
        )}
      </section>
      <footer className="composer-wrap">
        <form className="composer" onSubmit={send}>
          <textarea
            value={input}
            onChange={e=>setInput(e.target.value)}
            onKeyDown={e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send()}}}
            placeholder="Message ORBIT..."
            rows={1}
            disabled={busy}
          />
          <div className="composer-bottom">
            <span className="input-hint">Enter to send <span>·</span> Shift + Enter for a new line</span>
            <button type="submit" className="send-btn" disabled={!input.trim()||busy} aria-label="Send message">
              {busy?<LoaderCircle size={17} className="spin"/>:<ArrowUp size={17}/>}
            </button>
          </div>
        </form>
        <div className="footer-note">ORBIT can make mistakes. Verify important information.</div>
      </footer>
    </main>
  </div>
}
