import React, { useState } from "react";
import { useDropzone } from "react-dropzone";
import "./Classifier.css";
import { Spinner, Button, Alert, Image } from "react-bootstrap";
import axios from "axios";

export default function Classifier() {
  const [files, setFiles] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [recentImage, setRecentImage] = useState();

  const loadImage = (files) => {
    setTimeout(
      () => {
        setIsLoading(false);

        console.log(files[0].name);
      },

      1000
    );
  };

  const activateSpinner = () => {
    setFiles([]);
    setIsLoading(true);
  };

  const deactivateSpinner = () => {
    setIsLoading(false);
  };

  const getImageClass = (obj) => {
    axios
      .get(`http://127.0.0.1:8000/api/images/${obj.data.id}/`, {
        headers: {
          accept: "application/json",
        },
      })
      .then((resp) => {
        setRecentImage(resp);
        console.log(resp);
      })
      .catch((err) => {
        console.log(err);
      });
    deactivateSpinner();
  };

  const sendImage = () => {
    activateSpinner();
    let formData = new FormData();
    files[0] && formData.append("picture", files[0], files[0].name);
    axios
      .post("http://127.0.0.1:8000/api/images/", formData, {
        headers: {
          accept: "application/json",
          "content-type": "multipart/form-data",
        },
      })
      .then((resp) => {
        getImageClass(resp);
        console.log(resp.data.id);
      })
      .catch((err) => {
        console.log(err);
      });
  };

  const { isDragActive, getRootProps, getInputProps } = useDropzone({
    onDrop: (acceptedFiles) => {
      setFiles(acceptedFiles);
      setIsLoading(true);
      loadImage(acceptedFiles);
    },
  });

  return (
    <section className="container">
      <div {...getRootProps({ className: "dropzone back" })}>
        <input {...getInputProps()} />
        <p>
          {isDragActive
            ? "Drop some images"
            : "Drag 'n' drop some files here, or click to select files"}
        </p>
      </div>
      <aside>
        <ul>
          {files.map((file) => (
            <li key={file.path}>
              {file.path} - {file.size} bytes
            </li>
          ))}
        </ul>
        {files.length > 0 && (
          <Button variant="info" size="lg" className="mt-3" onClick={sendImage}>
            Select Image
          </Button>
        )}
        {isLoading && <Spinner animation="border" role="status"></Spinner>}
      </aside>
    </section>
  );
}
