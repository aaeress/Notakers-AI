import React, { useState, useEffect } from 'react';

function App() {
  const [text, setText] = useState('');
  const [realTimeText, setRealTimeText] = useState('');
  const [socket, setSocket] = useState(null);

  useEffect(() => {
    const newSocket = new WebSocket('ws://localhost:8000/ws');
    newSocket.onmessage = (event) => {
      setRealTimeText(event.data);
    };
    setSocket(newSocket);

    return () => {
      newSocket.close();
    };
  }, []);

  const handleInputChange = (event) => {
    const newText = event.target.value;
    setText(newText);
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(newText);
    }
  };

  const handleSubmit = async () => {
    console.log("Attempting to submit note:", text);

    // 构建表单数据
    const formData = new FormData();
    formData.append('text', text);

    try {
      const response = await fetch('http://localhost:8000/submit_note', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const data = await response.json();
      console.log('Note saved:', data);
    } catch (error) {
      console.error('Error saving note:', error);
      if (error.response) {
        console.log('Server response status:', error.response.status);
      } else if (error.request) {
        console.log('No response received:', error.request);
      } else {
        console.log('Error:', error.message);
      }
    }
  };

  return (
    <div className="App">
      <textarea
        value={text}
        onChange={handleInputChange}
        placeholder="Enter your notes here..."
        rows="10"
        cols="50"
      />
      <button onClick={handleSubmit}>Submit Note</button>
      <div>
        <h2>Real-time Text</h2>
        <p>{realTimeText}</p>
      </div>
    </div>
  );
}

export default App;
