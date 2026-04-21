import { useState } from 'react'

function App() {
  const [file, setFile] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [result, setResult] = useState(null)

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0])
    }
  }

  const handleUpload = async () => {
    if (!file) return

    setIsLoading(true)
    setResult(null)

    const formData = new FormData()
    formData.append('audio_file', file)

    try {
      const baseUrl = "https://shogo-toiyama--transcription-orchestrator-v1-fastapi-app.modal.run"

      const response = await fetch(`${baseUrl}/transcribe`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) throw new Error("アップロードに失敗しました")

      const data = await response.json()
      const jobId = data.job_id

      let attempts = 0
      const maxAttempts = 600

      const intervalId = setInterval(async () => {
        attempts += 1

        if (attempts > maxAttempts) {
          clearInterval(intervalId)
          setResult({ error: "タイムアウトしました。" })
          setIsLoading(false)
          return
        }

        try {
          const statusRes = await fetch(`${baseUrl}/status/${jobId}`)
          const statusData = await statusRes.json()

          if (statusData.status === 'completed') {
            clearInterval(intervalId)
            setResult(statusData.result)
            setIsLoading(false)
          } else if (statusData.status === 'error') {
            clearInterval(intervalId)
            setResult({ error: `サーバーエラー: ${statusData.message}` })
            setIsLoading(false)
          }
        } catch (pollError) {
          console.error("ポーリング中にエラー:", pollError)
        }
      }, 3000)

    } catch (error) {
      console.error('Upload failed:', error)
      setResult({ error: "処理中にエラーが発生しました。" })
      setIsLoading(false)
    }
  }

  return (
    <div
      style={{
        maxWidth: '900px',
        margin: '40px auto',
        fontFamily: 'sans-serif',
        padding: '0 20px'
      }}
    >
      <h1>🎤 文字起こし V1</h1>

      <div
        style={{
          marginBottom: '30px',
          padding: '20px',
          border: '1px solid #ccc',
          borderRadius: '8px'
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
            cursor: (!file || isLoading) ? 'not-allowed' : 'pointer',
            backgroundColor: isLoading ? '#999' : '#007BFF',
            color: 'white',
            border: 'none',
            borderRadius: '4px'
          }}
        >
          {isLoading ? '処理中...（数分かかる場合があります）' : 'アップロードして文字起こし'}
        </button>
      </div>

      {result && result.cleaned_text && (
        <div
          style={{
            background: '#f8f9fa',
            padding: '20px',
            borderRadius: '8px',
            border: '1px solid #e9ecef'
          }}
        >
          <h3 style={{ marginTop: 0, marginBottom: '16px' }}>整形後テキスト</h3>
          <pre
            style={{
              margin: 0,
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word',
              fontSize: '15px',
              lineHeight: '1.8',
              color: '#333',
              fontFamily: 'inherit'
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
            border: '1px solid #e9ecef'
          }}
        >
          <h3 style={{ marginTop: 0, marginBottom: '16px' }}>元の文字起こし</h3>
          <pre
            style={{
              margin: 0,
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word',
              fontSize: '15px',
              lineHeight: '1.8',
              color: '#333',
              fontFamily: 'inherit'
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

      {result && result.error && (
        <div
          style={{
            background: '#ffebee',
            padding: '20px',
            borderRadius: '8px',
            border: '1px solid #ffcdd2',
            color: '#c62828'
          }}
        >
          {result.error}
        </div>
      )}
    </div>
  )
}

export default App