import React, {useState} from 'react';
import { useRouter } from 'next/navigation';
import {DataTable} from 'primereact/datatable';
import {Column} from 'primereact/column';

import prisma from "../../prisma/client";
import {ProgressBar} from "primereact/progressbar";

export const getServerSideProps = async () => {
  const jobs = await prisma.job.findMany({
    take: 10,
    orderBy: {
      createdAt: 'desc'
    },
    include: {
      algorithm: true,
      dataset: true,
    }
  });

  return {
    props: {
      jobs: JSON.parse(JSON.stringify(jobs)),
    }
  }
}

export default function ListJobs({jobs}) {
  const router = useRouter()

  if (!jobs) {
    return <div>
      <h1 className="text-4xl font-bold">No jobs found</h1>
    </div>
  }

  const statusBodyTemplate = (rowData) => {
    let classNames = '';
    let progress = 0;
    if (rowData.status === 'completed') {
      progress = 100;
      classNames = 'p-progressbar-success';
    } else if (rowData.status === 'failed') {
      progress = 100;
      classNames = 'p-progressbar-failed';
    } else if (rowData.status === 'running') {
      progress = 30;
    }

    return (
      <React.Fragment>
        <ProgressBar value={progress} showValue={false} className={classNames}/>
      </React.Fragment>
    );
  };

  return <div>
    <h1 className="text-4xl font-bold">Jobs list</h1>

    <div className="card">
      <DataTable value={jobs} onRowClick={e => router.push(`/jobs/${e.data.id}`)} stripedRows size="small" rowClassName="p-selectable-row">
        <Column field="id" header="Job ID" body={(rowData) => <a href={`/jobs/${rowData.id}`}>{rowData.id}</a>}></Column>
        <Column field="status" header="Status" body={statusBodyTemplate}></Column>
        <Column field="dataset.name" header="Dataset"></Column>
        <Column field="algorithm.name" header="Algorithm"></Column>
        <Column field="scenario" header="Scenario"></Column>
        <Column field="recall" header="Recall"></Column>
        <Column field="epochs" header="Epochs"></Column>
        <Column field="createdAt" header="Created At"></Column>
      </DataTable>
    </div>
  </div>
}

