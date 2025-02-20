import React, {useState} from 'react';
import {useRouter} from 'next/navigation';
import {DataTable} from 'primereact/datatable';
import {Column} from 'primereact/column';
import {Button} from 'primereact/button';
import {RadioButton} from 'primereact/radiobutton';
import {InputNumber} from 'primereact/inputnumber';
import {Dropdown} from 'primereact/dropdown';
import {InputText} from "primereact/inputtext";

import prisma from "../prisma/client";
import {Checkbox} from "primereact/checkbox";

export const getServerSideProps = async ({req}) => {
  const algorithms = await prisma.algorithm.findMany();
  const datasets = await prisma.dataset.findMany();

  return {
    props: {
      algorithms: JSON.parse(JSON.stringify(algorithms)),
      datasets: JSON.parse(JSON.stringify(datasets)),
    }
  }
}

export default function Home({datasets, algorithms}) {
  const router = useRouter();

  const scenarios = [
    {code: 'filter', name: 'Filtering'},
    {code: 'verify', name: 'Verification'},
    {code: 'progress', name: 'Progressive'},
  ];

  // fields
  const [selectedScenario, setSelectedScenario] = useState(scenarios[0]);
  const [selectedDataset, setSelectedDataset] = useState(datasets[0]);
  const [selectedAlgorithm, setSelectedAlgorithm] = useState(algorithms.find(algo => algo.scenarios.includes(selectedScenario.code)));
  const [recall, setRecall] = useState(0.85);
  const [epochs, setEpochs] = useState(10);
  const [email, setEmail] = useState('');

  // state
  const [isLoading, setLoading] = useState(false);
  const [formDisabled, setDisabled] = useState(false);

  // data
  const [results, setResults] = useState([]);
  const [forceSubmit, setForceSubmit] = useState(false);
  const [job, setJob] = useState(null);

  const onSubmit = (e) => {
    setDisabled(true);

    fetch('/api/submit', {
      method: 'POST',
      headers: {
        'Accept': 'application/json',
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        scenario: selectedScenario.code,
        datasetId: selectedDataset.id,
        algorithmId: selectedAlgorithm.id,
        recall: recall,
        epochs: epochs,
        notifyEmail: email,
      })
    })
      .then((res) => {
        return res.json().then((data) => {
          console.log(res, data);

          if (res.status === 201) {
            setJob(data);
            router.push(`/jobs/${data.id}`);
          } else {
            setResults(data);
          }
        })
      });
  }

  if (isLoading) return <p>Loading...</p>
  if (!algorithms || !datasets) return <p>Something went wrong</p>

  return <div>
    <h1 className="text-4xl font-bold">No-code Benchmarking of Entity Resolution</h1>

    <h3 className="mb-2">Select a scenario</h3>
    <div className="flex flex-wrap gap-3">
      {scenarios.map((scenario) => {
        return (
          <div key={scenario.code} className="flex align-items-center">
            <RadioButton inputId={scenario.code} name="scenario" value={scenario}
                         onChange={(e) => setSelectedScenario(e.value)}
                         checked={selectedScenario.code === scenario.code} disabled={formDisabled}/>
            <label htmlFor={scenario.code} className="ml-2">{scenario.name}</label>
          </div>
        );
      })}
    </div>

    <div className="grid mt-4">
      <div className="col">
        <h2>Dataset</h2>

        <div className="flex flex-column gap-2 mb-3">
          <label htmlFor="dataset">Predefined dataset</label>
          <Dropdown id="dataset" aria-describedby="dataset-help"
                    value={selectedDataset} onChange={(e) => setSelectedDataset(e.value)}
                    disabled={formDisabled}
                    options={datasets} optionLabel="name"
                    placeholder="Select a dataset" className="w-full"/>
          <small id="dataset-help">You can select one of the common dataset, or upload your own
            below</small>
        </div>

        {/*<div className="flex flex-column gap-2 mb-3">
          <label htmlFor="epochs">Epochs</label>
          <InputNumber id="epochs" aria-describedby="epochs-help" className="w-full"
                       value={epochs} onValueChange={(e) => setEpochs(e.value)}
                       disabled={formDisabled}
                       minFractionDigits={0} min={5} max={50} step={5} showButtons/>
          <small id="epochs-help">A numbers of epochs to run</small>
        </div>*/}

        {/*<div className="flex flex-column gap-2 mb-3">
          <label htmlFor="dataset_file">Own dataset</label>
          <FileUpload id="dataset_file" aria-describedby="dataset_first-help" mode="basic"
                      name="dataset_file[]" accept="text/*" maxFileSize={1000000}
                      disabled={formDisabled}/>
          <small id="dataset_first-help">Please select the first part</small>

          <FileUpload id="dataset_second" aria-describedby="dataset_second-help" mode="basic"
                      name="dataset_file[]" accept="text/*" maxFileSize={1000000}
                      disabled={formDisabled}/>
          <small id="dataset_second-help">Please select the second part</small>

          <FileUpload id="dataset_ground" aria-describedby="dataset_ground-help" mode="basic"
                      name="dataset_file[]" accept="text/*" maxFileSize={1000000}
                      disabled={formDisabled}/>
          <small id="dataset_ground-help">Ground truth</small>
        </div>*/}
      </div>

      <div className="col">
        <h2>Model</h2>

        <div className="flex flex-column gap-2 mb-3">
          <label htmlFor="algorithm">Algorithm</label>
          <Dropdown id="algorithm" aria-describedby="algorithm-help"
                    value={selectedAlgorithm} onChange={(e) => setSelectedAlgorithm(e.value)}
                    disabled={formDisabled}
                    options={algorithms.filter(algo => algo.scenarios.includes(selectedScenario.code))}
                    optionLabel="name"
                    placeholder="Select a model" className="w-full"/>
          <small id="algorithm-help">Which algorithm / model to use</small>
        </div>

        <div className={(selectedAlgorithm != null && selectedAlgorithm.params.includes('recall') ? null : "hidden")}>
          <div className="flex flex-column gap-2 mb-3">
            <label htmlFor="recall">Recall</label>
            <InputNumber id="recall" aria-describedby="recall-help" className="w-full"
                         value={recall} onValueChange={(e) => setRecall(e.value)}
                         disabled={formDisabled}
                         minFractionDigits={2} min={0} max={1} step={0.05} mode="decimal"
                         showButtons/>
            <small id="recall-help">A recall value between 0 and 1</small>
          </div>
        </div>

        <div className={(selectedAlgorithm != null && selectedAlgorithm.params.includes('epochs') ? null : "hidden")}>
          <div className="flex flex-column gap-2 mb-3">
            <label htmlFor="epochs">Epochs</label>
            <InputNumber id="epochs" aria-describedby="epochs-help" className="w-full"
                         value={epochs} onValueChange={(e) => setEpochs(e.value)}
                         disabled={formDisabled}
                         minFractionDigits={0} min={5} max={50} step={5} showButtons/>
            <small id="epochs-help">A numbers of epochs to run</small>
          </div>
        </div>
      </div>
    </div>

    <div className="grid align-items-center">
      <div className="col">
        <div className="flex flex-column gap-2 mb-3">
          <label htmlFor="email">Email</label>
          <InputText id="email" aria-describedby="email-help" className="w-full"
                     value={email} onChange={(e) => setEmail(e.target.value)}
                     disabled={formDisabled}/>
          <small id="email-help">We will notify when training is complete, it may take some time</small>
        </div>
      </div>
    </div>

    <div className="grid">
      <div className="col-auto">
        <Button label="Submit" onClick={onSubmit} disabled={results.length > 0 && !forceSubmit}/>
      </div>
      {results.length > 0 && (
        <div className="col">
          <Checkbox inputId="force" onChange={e => setForceSubmit(e.checked)} checked={forceSubmit}></Checkbox>
          <label htmlFor="force" className="p-checkbox-label pl-2">I want to submit a new job</label>
        </div>
      )}
    </div>

    {results.length > 0 && (
      <div className="grid">
        <div className="col">
          <h4>We have found some similar results already computed</h4>

          <DataTable value={results} onRowClick={e => router.push(`/jobs/${e.data.id}`)} stripedRows size="small" rowClassName="p-selectable-row">
            <Column field="id" header="Job ID" body={(rowData) => <a href={`/jobs/${rowData.id}`}>{rowData.id}</a>}></Column>
            <Column field="dataset.name" header="Dataset"></Column>
            <Column field="algorithm.name" header="Algorithm"></Column>
            <Column field="scenario" header="Scenario"></Column>
            <Column field="recall" header="Recall"></Column>
            <Column field="epochs" header="Epochs"></Column>
            <Column field="createdAt" header="Created At"></Column>
          </DataTable>
        </div>
      </div>
    )}
  </div>
}

