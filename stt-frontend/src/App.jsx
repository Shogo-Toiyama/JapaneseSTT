import { useState, useMemo } from 'react'

function App() {
  const [file, setFile] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [elapsedSec, setElapsedSec] = useState(0)
  const [activeTab, setActiveTab] = useState('final')

  const [statusPayload, setStatusPayload] = useState(null)
  const [debugData, setDebugData] = useState(null)
  const [selectedAsrChunk, setSelectedAsrChunk] = useState(0)
  const [selectedLlmInputChunk, setSelectedLlmInputChunk] = useState(0)
  const [selectedLlmOutputChunk, setSelectedLlmOutputChunk] = useState(0)

  const debugTabs = useMemo(
    () => [
      { key: 'final', label: 'Final' },
      { key: 'asr', label: 'ASR' },
      { key: 'diarization', label: 'Diarization' },
      { key: 'fusion', label: 'Fusion' },
      { key: 'llmInput', label: 'LLM Input' },
      { key: 'llmOutput', label: 'LLM Output' },
    ],
    []
  )

  const formatElapsed = (sec) => {
    if (sec == null || Number.isNaN(sec)) return '-'
    if (sec < 60) return `${sec.toFixed(1)} 秒`

    const minutes = Math.floor(sec / 60)
    const seconds = sec % 60
    return `${minutes} 分 ${seconds.toFixed(1)} 秒`
  }

  const safeJson = (value) => JSON.stringify(value ?? null, null, 2)

  const getAsrChunks = () => debugData?.asr?.chunks ?? []
  const getSpeakerSegments = () => debugData?.diarization?.speaker_segments ?? []
  const getFusionTranscript = () => debugData?.fusion?.transcript ?? []
  const getLlmInputs = () => debugData?.llm?.inputs ?? []
  const getLlmOutputs = () => debugData?.llm?.outputs ?? []

  const selectedAsrChunkData = getAsrChunks()[selectedAsrChunk] ?? null
  const selectedLlmInputData = getLlmInputs()[selectedLlmInputChunk] ?? null
  const selectedLlmOutputData = getLlmOutputs()[selectedLlmOutputChunk] ?? null

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0])
    }
  }

  const handleUpload = async () => {
    if (!file) return

    setIsLoading(true)
    setResult(null)
    setElapsedSec(0)
    setActiveTab('final')
    setStatusPayload(null)
    setDebugData(null)
    setSelectedAsrChunk(0)
    setSelectedLlmInputChunk(0)
    setSelectedLlmOutputChunk(0)

    const formData = new FormData()
    formData.append('audio_file', file)

    try {
      const baseUrl =
        'https://shogo-toiyama--transcription-orchestrator-v1-fastapi-app.modal.run'

      const response = await fetch(`${baseUrl}/transcribe`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) throw new Error('アップロードに失敗しました')

      const data = await response.json()
      const jobId = data.job_id

      let attempts = 0
      const maxAttempts = 600

      const intervalId = setInterval(async () => {
        attempts += 1

        if (attempts > maxAttempts) {
          clearInterval(intervalId)
          setResult({
            error: 'タイムアウトしました。',
            timing: {
              elapsed_sec: elapsedSec,
            },
          })
          setIsLoading(false)
          return
        }

        try {
          const statusRes = await fetch(`${baseUrl}/status/${jobId}`)
          const statusData = await statusRes.json()

          setStatusPayload(statusData)
          setDebugData(statusData?.result?.debug ?? statusData?.debug ?? null)

          if (typeof statusData.current_elapsed_sec === 'number') {
            setElapsedSec(statusData.current_elapsed_sec)
          }

          if (statusData.status === 'completed') {
            clearInterval(intervalId)

            const finalResult = {
              ...statusData.result,
              timing: statusData.result?.timing ?? {
                elapsed_sec: statusData.elapsed_sec,
                created_at: statusData.created_at,
                started_at: statusData.started_at,
                finished_at: statusData.finished_at,
              },
              debug: statusData?.result?.debug ?? statusData?.debug ?? null,
            }

            setResult(finalResult)
            setDebugData(finalResult.debug ?? null)
            setElapsedSec(statusData.elapsed_sec ?? statusData.current_elapsed_sec ?? 0)
            setIsLoading(false)
          } else if (statusData.status === 'error') {
            clearInterval(intervalId)

            setResult({
              error: `サーバーエラー: ${statusData.message}`,
              timing: {
                elapsed_sec: statusData.elapsed_sec,
              },
              debug: statusData?.debug ?? null,
            })
            setDebugData(statusData?.debug ?? null)
            setElapsedSec(statusData.elapsed_sec ?? statusData.current_elapsed_sec ?? 0)
            setIsLoading(false)
          }
        } catch (pollError) {
          console.error('ポーリング中にエラー:', pollError)
        }
      }, 3000)
    } catch (error) {
      console.error('Upload failed:', error)
      setResult({
        error: '処理中にエラーが発生しました。',
      })
      setIsLoading(false)
    }
  }

  const Panel = ({ title, children, subtitle }) => (
    <div
      style={{
        background: '#fff',
        border: '1px solid #e5e7eb',
        borderRadius: '10px',
        overflow: 'hidden',
        marginBottom: '16px',
      }}
    >
      <div
        style={{
          padding: '12px 16px',
          borderBottom: '1px solid #e5e7eb',
          background: '#fafafa',
        }}
      >
        <div style={{ fontWeight: 600 }}>{title}</div>
        {subtitle ? (
          <div style={{ marginTop: '4px', fontSize: '13px', color: '#666' }}>
            {subtitle}
          </div>
        ) : null}
      </div>
      <div style={{ padding: '16px' }}>{children}</div>
    </div>
  )
  
  const CodeBlock = ({ value, maxHeight = 420 }) => (
    <pre
      style={{
        margin: 0,
        whiteSpace: 'pre-wrap',
        wordBreak: 'break-word',
        fontSize: '13px',
        lineHeight: '1.6',
        color: '#222',
        background: '#f8f9fa',
        border: '1px solid #e5e7eb',
        borderRadius: '8px',
        padding: '12px',
        maxHeight,
        overflow: 'auto',
        fontFamily:
          'ui-monospace, SFMono-Regular, SF Mono, Menlo, Consolas, monospace',
      }}
    >
      {value}
    </pre>
  )
  
  const SelectList = ({ items, selectedIndex, onSelect, renderLabel }) => (
    <div
      style={{
        border: '1px solid #e5e7eb',
        borderRadius: '8px',
        overflow: 'auto',
        maxHeight: '420px',
        background: '#fff',
      }}
    >
      {items.length === 0 ? (
        <div style={{ padding: '12px', color: '#666' }}>データがありません。</div>
      ) : (
        items.map((item, index) => (
          <button
            key={index}
            onClick={() => onSelect(index)}
            style={{
              display: 'block',
              width: '100%',
              textAlign: 'left',
              padding: '10px 12px',
              border: 'none',
              borderBottom: index !== items.length - 1 ? '1px solid #f0f0f0' : 'none',
              background: selectedIndex === index ? '#EAF3FF' : '#fff',
              cursor: 'pointer',
              fontSize: '13px',
            }}
          >
            {renderLabel(item, index)}
          </button>
        ))
      )}
    </div>
  )

  const renderDebugTabContent = () => {
    switch (activeTab) {
      case 'final':
        return (
          <>
            <Panel
              title="Final Text"
              subtitle="LLM整形後の最終テキスト"
            >
              <CodeBlock
                value={
                  result?.cleaned_text ||
                  'cleaned_text はまだありません。'
                }
                maxHeight={520}
              />
            </Panel>
  
            <Panel
              title="Final Transcript JSON"
              subtitle={`items: ${result?.cleaned_transcript?.length ?? 0}`}
            >
              <CodeBlock value={safeJson(result?.cleaned_transcript ?? [])} />
            </Panel>
          </>
        )
  
      case 'asr': {
        const asrChunks = getAsrChunks()
        const chunk = selectedAsrChunkData
  
        return (
          <>
            <Panel
              title="ASR Summary"
              subtitle="ASR chunkごとの結果と char_timestamps"
            >
              <CodeBlock value={safeJson(debugData?.asr?.summary ?? {})} maxHeight={220} />
            </Panel>
  
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: '320px 1fr',
                gap: '16px',
              }}
            >
              <Panel
                title="ASR Chunks"
                subtitle={`chunks: ${asrChunks.length}`}
              >
                <SelectList
                  items={asrChunks}
                  selectedIndex={selectedAsrChunk}
                  onSelect={setSelectedAsrChunk}
                  renderLabel={(item, index) =>
                    `#${index} [${item?.start ?? '-'} - ${item?.end ?? '-'}] ${
                      (item?.text ?? '').slice(0, 40) || '(empty)'
                    }`
                  }
                />
              </Panel>
  
              <div>
                <Panel title="Selected ASR Chunk">
                  <CodeBlock value={safeJson(chunk)} maxHeight={220} />
                </Panel>
  
                <Panel title="Selected ASR Text">
                  <CodeBlock value={chunk?.text || '(empty)'} maxHeight={180} />
                </Panel>
  
                <Panel title="Selected ASR Char Timestamps">
                  <CodeBlock value={safeJson(chunk?.char_timestamps ?? [])} maxHeight={520} />
                </Panel>
              </div>
            </div>
          </>
        )
      }
  
      case 'diarization':
        return (
          <>
            <Panel
              title="Diarization Summary"
              subtitle="話者分離結果の概要"
            >
              <CodeBlock value={safeJson(debugData?.diarization?.summary ?? {})} maxHeight={220} />
            </Panel>
  
            <Panel
              title="Speaker Segments"
              subtitle={`segments: ${getSpeakerSegments().length}`}
            >
              <CodeBlock value={safeJson(getSpeakerSegments())} maxHeight={560} />
            </Panel>
          </>
        )
  
      case 'fusion':
        return (
          <>
            <Panel
              title="Fusion Summary"
              subtitle="ASR + speaker を結合した中間結果"
            >
              <CodeBlock value={safeJson(debugData?.fusion?.summary ?? {})} maxHeight={220} />
            </Panel>
  
            <Panel title="Fusion Plain Text">
              <CodeBlock value={debugData?.fusion?.plain_text || '(empty)'} maxHeight={420} />
            </Panel>
  
            <Panel
              title="Fusion Transcript JSON"
              subtitle={`items: ${getFusionTranscript().length}`}
            >
              <CodeBlock value={safeJson(getFusionTranscript())} maxHeight={560} />
            </Panel>
          </>
        )
  
      case 'llmInput': {
        const llmInputs = getLlmInputs()
        const input = selectedLlmInputData
  
        return (
          <>
            <Panel
              title="LLM Input Summary"
              subtitle="LLMに投げた Context / Target"
            >
              <CodeBlock value={safeJson(debugData?.llm?.summary ?? {})} maxHeight={220} />
            </Panel>
  
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: '320px 1fr',
                gap: '16px',
              }}
            >
              <Panel
                title="LLM Input Chunks"
                subtitle={`chunks: ${llmInputs.length}`}
              >
                <SelectList
                  items={llmInputs}
                  selectedIndex={selectedLlmInputChunk}
                  onSelect={setSelectedLlmInputChunk}
                  renderLabel={(item, index) =>
                    `#${index + 1} prompt_chars=${item?.prompt_chars ?? 0}`
                  }
                />
              </Panel>
  
              <div>
                <Panel title="Selected Input Metadata">
                  <CodeBlock value={safeJson(input)} maxHeight={220} />
                </Panel>
  
                <Panel title="Context Lines">
                  <CodeBlock value={(input?.context_lines ?? []).join('\n')} maxHeight={220} />
                </Panel>
  
                <Panel title="Target Lines">
                  <CodeBlock value={(input?.target_lines ?? []).join('\n')} maxHeight={420} />
                </Panel>
              </div>
            </div>
          </>
        )
      }
  
      case 'llmOutput': {
        const llmOutputs = getLlmOutputs()
        const output = selectedLlmOutputData
  
        return (
          <>
            <Panel
              title="LLM Output Summary"
              subtitle="LLMから返ってきた chunk ごとの生テキスト"
            >
              <CodeBlock value={safeJson(debugData?.llm?.summary ?? {})} maxHeight={220} />
            </Panel>
  
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: '320px 1fr',
                gap: '16px',
              }}
            >
              <Panel
                title="LLM Output Chunks"
                subtitle={`chunks: ${llmOutputs.length}`}
              >
                <SelectList
                  items={llmOutputs}
                  selectedIndex={selectedLlmOutputChunk}
                  onSelect={setSelectedLlmOutputChunk}
                  renderLabel={(item, index) =>
                    `#${index + 1} chars=${item?.response_chars ?? 0} time=${item?.elapsed_sec ?? '-'}s`
                  }
                />
              </Panel>
  
              <div>
                <Panel title="Selected Output Metadata">
                  <CodeBlock value={safeJson(output)} maxHeight={220} />
                </Panel>
  
                <Panel title="Selected Output Raw Text">
                  <CodeBlock value={output?.raw_text || '(empty)'} maxHeight={520} />
                </Panel>
              </div>
            </div>
          </>
        )
      }
  
      default:
        return null
    }
  }

  return (
    <div
      style={{
        maxWidth: '960px',
        margin: '40px auto',
        fontFamily: 'sans-serif',
        padding: '0 20px',
      }}
    >
      <h1>🎤 文字起こし V1</h1>

      <div
        style={{
          marginBottom: '30px',
          padding: '20px',
          border: '1px solid #ccc',
          borderRadius: '8px',
          background: '#fff',
        }}
      >
        <input
          type="file"
          accept="audio/*"
          onChange={handleFileChange}
          style={{ marginBottom: '15px', display: 'block' }}
        />

        <button
          onClick={handleUpload}
          disabled={!file || isLoading}
          style={{
            padding: '10px 20px',
            fontSize: '16px',
            cursor: !file || isLoading ? 'not-allowed' : 'pointer',
            backgroundColor: isLoading ? '#999' : '#007BFF',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
          }}
        >
          {isLoading ? '処理中...' : 'アップロードして文字起こし'}
        </button>

        {isLoading && (
          <div style={{ marginTop: '12px', color: '#555', fontSize: '14px' }}>
            処理経過時間: {formatElapsed(elapsedSec)}
          </div>
        )}
      </div>

      {result && result.cleaned_text && (
        <div
          style={{
            background: '#f8f9fa',
            padding: '20px',
            borderRadius: '8px',
            border: '1px solid #e9ecef',
            marginBottom: '24px',
          }}
        >
          <h3 style={{ marginTop: 0, marginBottom: '8px' }}>処理結果</h3>
          {result?.timing?.elapsed_sec != null && (
            <div style={{ marginBottom: '16px', color: '#666', fontSize: '14px' }}>
              総処理時間: {formatElapsed(Number(result.timing.elapsed_sec))}
            </div>
          )}

          <pre
            style={{
              margin: 0,
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word',
              fontSize: '15px',
              lineHeight: '1.8',
              color: '#333',
              fontFamily: 'inherit',
            }}
          >
            {result.cleaned_text}
          </pre>
        </div>
      )}

      {!result?.cleaned_text && result?.transcript && (
        <div
          style={{
            background: '#f8f9fa',
            padding: '20px',
            borderRadius: '8px',
            border: '1px solid #e9ecef',
            marginBottom: '24px',
          }}
        >
          <h3 style={{ marginTop: 0, marginBottom: '8px' }}>元の文字起こし</h3>
          {result?.timing?.elapsed_sec != null && (
            <div style={{ marginBottom: '16px', color: '#666', fontSize: '14px' }}>
              総処理時間: {formatElapsed(Number(result.timing.elapsed_sec))}
            </div>
          )}

          <pre
            style={{
              margin: 0,
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word',
              fontSize: '15px',
              lineHeight: '1.8',
              color: '#333',
              fontFamily: 'inherit',
            }}
          >
            {result.transcript
              .map(
                (item) =>
                  `[${item.start.toFixed(1)}s] ${item.speaker}: ${item.text}`
              )
              .join('\n')}
          </pre>
        </div>
      )}

      {(result || debugData || statusPayload) && (
        <div
          style={{
            background: '#fff',
            border: '1px solid #e5e7eb',
            borderRadius: '10px',
            overflow: 'hidden',
            marginBottom: '24px',
          }}
        >
          <div
            style={{
              padding: '16px 20px',
              borderBottom: '1px solid #e5e7eb',
              background: '#fafafa',
            }}
          >
            <h2 style={{ margin: 0, fontSize: '18px' }}>Debug Viewer</h2>
            <p style={{ margin: '6px 0 0', color: '#666', fontSize: '14px' }}>
              ASR / Diarization / Fusion / LLM の中間結果を確認できます。
            </p>
          </div>

          <div
            style={{
              display: 'flex',
              flexWrap: 'wrap',
              gap: '8px',
              padding: '12px',
              borderBottom: '1px solid #e5e7eb',
              background: '#f8f9fa',
            }}
          >
            {debugTabs.map((tab) => (
              <button
                key={tab.key}
                onClick={() => setActiveTab(tab.key)}
                style={{
                  padding: '8px 12px',
                  borderRadius: '8px',
                  border:
                    activeTab === tab.key
                      ? '1px solid #007BFF'
                      : '1px solid #d0d7de',
                  background: activeTab === tab.key ? '#EAF3FF' : '#fff',
                  color: '#222',
                  cursor: 'pointer',
                  fontSize: '14px',
                }}
              >
                {tab.label}
              </button>
            ))}
          </div>

          <div
            style={{
              display: 'flex',
              flexWrap: 'wrap',
              gap: '12px',
              padding: '12px 16px',
              borderBottom: '1px solid #e5e7eb',
              background: '#fff',
              fontSize: '13px',
              color: '#555',
            }}
          >
            <div>status: {statusPayload?.status ?? '-'}</div>
            <div>ASR chunks: {debugData?.asr?.summary?.chunk_count ?? 0}</div>
            <div>
              speaker segments: {debugData?.diarization?.summary?.speaker_segment_count ?? 0}
            </div>
            <div>fusion items: {debugData?.fusion?.summary?.transcript_items ?? 0}</div>
            <div>LLM inputs: {debugData?.llm?.summary?.input_chunk_count ?? 0}</div>
            <div>LLM outputs: {debugData?.llm?.summary?.output_chunk_count ?? 0}</div>
          </div>

          <div style={{ padding: '20px' }}>{renderDebugTabContent()}</div>
        </div>
      )}

      {result && result.error && (
        <div
          style={{
            background: '#ffebee',
            padding: '20px',
            borderRadius: '8px',
            border: '1px solid #ffcdd2',
            color: '#c62828',
          }}
        >
          <div>{result.error}</div>
          {result?.timing?.elapsed_sec != null && (
            <div style={{ marginTop: '8px', fontSize: '14px' }}>
              経過時間: {formatElapsed(Number(result.timing.elapsed_sec))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default App